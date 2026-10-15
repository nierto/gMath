//! The sure path of the binary transcendentals: `exp`, `ln` and the
//! functions built on them, evaluated at about 1000 fraction bits on I1024
//! and rounded once to the storage format.
//!
//! The format engines carry an error relative to the result and leave as
//! few as zero guard bits near the top of a format's range (the compute
//! tier has exactly twice the storage's fraction bits). The imperative
//! functions call the engine first and come here only when the engine's
//! result lies too close to a rounding boundary to decide it; this path then
//! decides it with hundreds of bits to spare, for every format including
//! Q256.256.
//!
//! Method. `exp(x) = 2^n * exp(r)` with `x = n ln2 + r`, `0 <= r < ln2`;
//! `exp(r) = exp(r / 2^K)^(2^K)` with the Taylor series of the tiny argument
//! and `K` squarings. `ln(v)` by Newton on `exp` from a 60-bit start.
//! `ln 2 = 2 artanh(1/3)`, computed once per thread. Integer arithmetic only.

use crate::fixed_point::width::WideInt;
use crate::fixed_point::{I1024, I2048};

/// Fraction bits of the working values.
pub(crate) const P: u32 = 1000;
/// Halvings of the reduced argument before the Taylor series.
const K: u32 = 40;

#[inline]
pub(crate) fn one() -> I1024 { I1024::from_i128(1).shl_bits(P) }

/// `a * b` at P fraction bits, any signs, truncated toward zero.
pub(crate) fn mul(a: I1024, b: I1024) -> I1024 {
    let neg = a.is_negative() != b.is_negative();
    let (ma, mb) = (if a.is_negative() { -a } else { a }, if b.is_negative() { -b } else { b });
    let p = (ma.mul_to_i2048(mb) >> P as usize).as_i1024();
    if neg { -p } else { p }
}

/// `a / d` for a small positive divisor, any sign of `a`, truncated toward zero.
pub(crate) fn div_small(a: I1024, d: u64) -> I1024 {
    let neg = a.is_negative();
    let mut m = if neg { -a } else { a };
    let mut rem: u128 = 0;
    for i in (0..16).rev() {
        let cur = (rem << 64) | m.words[i] as u128;
        m.words[i] = (cur / d as u128) as u64;
        rem = cur % d as u128;
    }
    if neg { -m } else { m }
}

/// `a * n` for a small integer, any signs.
pub(crate) fn mul_small(a: I1024, n: i64) -> I1024 {
    let neg = (n < 0) != a.is_negative();
    let ma = if a.is_negative() { -a } else { a };
    let p = (ma.mul_to_i2048(I1024::from_i128(n.unsigned_abs() as i128))).as_i1024();
    if neg { -p } else { p }
}

/// ln 2 at P fraction bits: `2 * sum (1/3)^(2k+1) / (2k+1)`.
pub(crate) fn ln2() -> I1024 {
    thread_local! {
        static LN2: I1024 = {
            let mut t = div_small(one(), 3);
            let mut sum = I1024::from_i128(0);
            let mut k: u64 = 0;
            while t != I1024::from_i128(0) {
                sum = sum + div_small(t, 2 * k + 1);
                t = div_small(t, 9);
                k += 1;
            }
            sum + sum
        };
    }
    LN2.with(|v| *v)
}

/// `exp(x)` for `x` at P fraction bits, as `(m, n)` with `exp(x) = m * 2^n`
/// and `m` in `[1, 2)` at P fraction bits.
pub fn exp_parts(x: I1024) -> (I1024, i64) {
    let l2 = ln2();
    // n from a 64-bit view of both, then corrected exactly
    let xs = x.shr_bits(P - 60);
    let x_hi = i128_of(xs);
    let l2_hi = i128_of(l2.shr_bits(P - 60));
    let mut n = x_hi.div_euclid(l2_hi) as i64;
    let mut r = x - mul_small(l2, n);
    while r.is_negative() { n -= 1; r = r + l2; }
    while !(r < l2) { n += 1; r = r - l2; }
    // exp(r) = exp(r / 2^K)^(2^K)
    let s = r.shr_bits(K);
    let mut sum = one();
    let mut term = one();
    let mut k: u64 = 1;
    loop {
        term = div_small(mul(term, s), k);
        if term == I1024::from_i128(0) { break; }
        sum = sum + term;
        k += 1;
    }
    for _ in 0..K { sum = mul(sum, sum); }
    // exp(r) is in [1, 2) for r in [0, ln 2); the rounding of the squarings
    // can leave it a hair outside
    if !(sum < one().shl_bits(1)) { return (sum.shr_bits(1), n + 1); }
    (sum, n)
}

/// `ln(v)` at P fraction bits for `v = raw * 2^-f > 0`.
pub fn ln_of_raw<S: WideInt>(raw: S, f: u32) -> I1024 {
    let raw = to_i1024(raw);
    // v = m * 2^e with m in [1, 2)
    let e = raw.bit_length() as i64 - 1 - f as i64;
    let top = raw.bit_length() - 1;
    let m = if top <= P { raw.shl_bits(P - top) } else { raw.shr_bits(top - P) };
    let l2 = ln2();
    // ln m by Newton on exp: y <- y + m * exp(-y) - 1, from a 60-bit start
    let m60 = i128_of(m.shr_bits(P - 60));
    let mut y = {
        // ln at Q64.64 of m (in [1, 2)): the format engine, as a start only
        let q64 = crate::fixed_point::domains::binary_fixed::transcendental::ln_tier_n_plus_1::ln_q64_64_native(m60 << 4);
        I1024::from_i128(q64).shl_bits(P - 64)
    };
    for _ in 0..6 {
        let (em, en) = exp_parts(-y);
        let prod = mul(m, em);
        let prod = if en >= 0 { prod.shl_bits(en as u32) } else { prod.shr_bits((-en) as u32) };
        y = y + prod - one();
    }
    y + mul_small(l2, e)
}

/// `exp(x)` rounded to the nearest raw at `f` fraction bits (ties toward
/// +infinity), `None` beyond the word `S`. `x` at P fraction bits.
pub fn exp_to_raw<S: WideInt>(x: I1024, f: u32) -> Option<S> {
    let (m, n) = exp_parts(x);
    round_scaled::<S>(m, n, f)
}

/// `m * 2^n` (m at P fraction bits, non-negative) rounded to a raw at `f`.
fn round_scaled<S: WideInt>(m: I1024, n: i64, f: u32) -> Option<S> {
    let shift = P as i64 - n - f as i64;
    if shift <= 0 { return None; }
    if shift > 1020 { return Some(S::from_i128(0)); }
    let s = shift as u32;
    let up = m.bit(s - 1);
    let q = m.shr_bits(s) + if up { I1024::from_i128(1) } else { I1024::from_i128(0) };
    from_i1024::<S>(q)
}

/// A signed value at P fraction bits rounded to a raw at `f` (nearest, ties
/// toward +infinity).
pub fn round_signed<S: WideInt>(v: I1024, f: u32) -> Option<S> {
    let s = P - f;
    let q = v.shr_bits(s) + if v.bit(s - 1) { I1024::from_i128(1) } else { I1024::from_i128(0) };
    from_i1024::<S>(q)
}

/// A raw at `f` fraction bits as a value at P fraction bits.
pub fn from_raw<S: WideInt>(raw: S, f: u32) -> I1024 {
    debug_assert!(int_bits_of(raw, f) <= 20, "sure path: argument too large for P bits");
    to_i1024(raw).shl_bits(P - f)
}

/// Bits of the integer part of |raw * 2^-f|.
fn int_bits_of<S: WideInt>(raw: S, f: u32) -> u32 {
    let m = to_i1024(raw);
    let m = if m.is_negative() { -m } else { m };
    m.bit_length().saturating_sub(f)
}

/// `exp` of a raw at `f`: `None` beyond the word, 0 far below it.
pub fn exp_raw_to_raw<S: WideInt>(raw: S, f: u32) -> Option<S> {
    if int_bits_of(raw, f) > 20 {
        // |x| >= 2^20: e^x overflows every word, e^-x underflows it
        return if raw.is_negative() { Some(S::from_i128(0)) } else { None };
    }
    exp_to_raw::<S>(from_raw(raw, f), f)
}

/// `1/x` at P fraction bits for a raw at `f`, |x| >= 1.
fn inv_of_raw<S: WideInt>(raw: S, f: u32) -> I1024 {
    let num = I2048::from_i128(1).shl_bits(P + f);
    let (q, _) = crate::fixed_point::domains::binary_fixed::i2048::i2048_divmod(num, to_i2048(raw));
    q.as_i1024()
}

/// `sinh` / `cosh` of a raw at `f`, rounded once: `(e^x -/+ e^-x) / 2`.
pub fn sinh_cosh_to_raw<S: WideInt>(x_raw: S, f: u32, cosh: bool) -> Option<S> {
    if int_bits_of(x_raw, f) > 20 { return None; }
    let x = from_raw(x_raw, f);
    let (a, an) = exp_parts(x);
    let (b, bn) = exp_parts(-x);
    // both at a common exponent: e^|x| dominates; bring the small one down
    let (big, bign, small, smalln) = if an >= bn { (a, an, b, bn) } else { (b, bn, a, an) };
    let small = small.shr_bits((bign - smalln).min(1020) as u32);
    let neg_side = !(an >= bn); // x < 0: e^x is the small one
    let v = if cosh { big + small } else if neg_side { small - big } else { big - small };
    // v * 2^(bign - 1), v may be negative for sinh of a negative argument
    if v.is_negative() {
        let r: S = round_scaled_signed(-v, bign - 1, f, true)?;
        return Some(r);
    }
    round_scaled::<S>(v, bign - 1, f)
}

/// Rounding of `-(m * 2^n)` with ties toward +infinity: the nearest of the
/// negated magnitude, with an exact tie going up (toward zero).
fn round_scaled_signed<S: WideInt>(m: I1024, n: i64, f: u32, negative: bool) -> Option<S> {
    let shift = P as i64 - n - f as i64;
    if shift <= 0 { return None; }
    if shift > 1020 { return Some(S::from_i128(0)); }
    let s = shift as u32;
    let fl = m.shr_bits(s);
    let rem_any_below = m.low_bits_any(s - 1);
    let half = m.bit(s - 1);
    // magnitude rounded: half with nothing below is a tie -> toward +inf = smaller magnitude when negative
    let up = if negative { half && rem_any_below } else { half };
    let q = fl + if up { I1024::from_i128(1) } else { I1024::from_i128(0) };
    from_i1024::<S>(if negative { -q } else { q })
}

/// `x^y` for raws at `f` (x > 0), rounded once.
pub fn pow_to_raw<S: WideInt>(x_raw: S, y_raw: S, f: u32) -> Option<S> {
    let l = ln_of_raw(x_raw, f);
    // t = y * ln x, y = y_raw * 2^-f exact: the product on I2048
    let y = to_i1024(y_raw);
    let neg = l.is_negative() != y.is_negative();
    let (ml, my) = (if l.is_negative() { -l } else { l }, if y.is_negative() { -y } else { y });
    let p: I2048 = ml.mul_to_i2048(my) >> f as usize;
    if p.bit_length() > 1012 { return None; }
    let t = p.as_i1024();
    exp_to_raw::<S>(if neg { -t } else { t }, f)
}

fn i128_of(v: I1024) -> i128 {
    let neg = v.is_negative();
    let m = if neg { -v } else { v };
    let lo = (m.words[0] as u128 | ((m.words[1] as u128) << 64)) as i128;
    if neg { -lo } else { lo }
}

pub(crate) fn to_i1024<S: WideInt>(raw: S) -> I1024 {
    let neg = raw.is_negative();
    let m = if neg { raw.checked_neg_wide() } else { Some(raw) };
    let mut out = I1024::from_i128(0);
    match m {
        Some(m) => {
            for i in 0..m.bit_length() { if m.bit(i) { out.words[(i / 64) as usize] |= 1u64 << (i % 64); } }
            if neg { -out } else { out }
        }
        None => {
            // the word's minimum: -2^(BITS-1)
            -I1024::from_i128(1).shl_bits(S::BITS - 1)
        }
    }
}

fn from_i1024<S: WideInt>(v: I1024) -> Option<S> {
    let neg = v.is_negative();
    let m = if neg { -v } else { v };
    let n = m.bit_length();
    if n > S::BITS - 1 {
        // only -2^(BITS-1) fits at this length
        if neg && n == S::BITS && m == I1024::from_i128(1).shl_bits(S::BITS - 1) {
            return Some(S::from_i128(1).shl_bits(S::BITS - 1));
        }
        return None;
    }
    let mut out = S::from_i128(0);
    for i in 0..n { if m.bit(i) { out = out.checked_add_wide(S::from_i128(1).shl_bits(i)).expect("fits"); } }
    Some(if neg { out.checked_neg_wide().expect("fits") } else { out })
}


// ============================================================================
// Trigonometric and inverse-hyperbolic sure paths
// ============================================================================

/// Fraction bits of the I2048 values used for argument reduction: the
/// argument (up to 2^255 at Q256.256) times 2^PB fits I2048.
pub(crate) const PB: u32 = 1500;

fn div_small_2048(a: I2048, d: u64) -> I2048 {
    let neg = a.is_negative();
    let mut m = if neg { -a } else { a };
    let mut rem: u128 = 0;
    for i in (0..32).rev() {
        let cur = (rem << 64) | m.words[i] as u128;
        m.words[i] = (cur / d as u128) as u64;
        rem = cur % d as u128;
    }
    if neg { -m } else { m }
}

/// pi/2 at PB fraction bits, Machin: pi = 16 atan(1/5) - 4 atan(1/239).
pub(crate) fn half_pi_2048() -> I2048 {
    thread_local! {
        static HP: I2048 = {
            fn atan_inv(n: u64) -> I2048 {
                let one = I2048::from_i128(1).shl_bits(PB);
                let mut t = div_small_2048(one, n);
                let n2 = n * n;
                let mut sum = I2048::from_i128(0);
                let mut k: u64 = 0;
                while t != I2048::from_i128(0) {
                    let term = div_small_2048(t, 2 * k + 1);
                    sum = if k % 2 == 0 { sum + term } else { sum - term };
                    t = div_small_2048(t, n2);
                    k += 1;
                }
                sum
            }
            let a = atan_inv(5);
            let b = atan_inv(239);
            // pi/2 = 8 atan(1/5) - 2 atan(1/239)
            a.shl_bits(3) - b.shl_bits(1)
        };
    }
    HP.with(|v| *v)
}

/// `x = q * pi/2 + r`, `0 <= r < pi/2`: `(r at P fraction bits, q mod 4)`.
fn reduce_half_pi<S: WideInt>(raw: S, f: u32) -> (I1024, u32) {
    reduce_half_pi_wide(to_i2048(raw).shl_bits(PB - f))
}

/// `x = q * pi/2 + r` for `x` at PB fraction bits on I2048: `(r at P
/// fraction bits, q mod 4)`.
pub(crate) fn reduce_half_pi_wide(x: I2048) -> (I1024, u32) {
    let hp = half_pi_2048();
    let (mut q, mut r) = crate::fixed_point::domains::binary_fixed::i2048::i2048_divmod(x, hp);
    if r.is_negative() { r = r + hp; q = q - I2048::from_i128(1); }
    // two's complement low bits are q mod 4 for either sign
    ((r >> (PB - P) as usize).as_i1024(), (q.words[0] & 3) as u32)
}

/// `(sin r, cos r)` at P fraction bits for `0 <= r < 2`: Taylor of `r/2^20`,
/// then twenty double-angle steps.
pub(crate) fn sincos_reduced(r: I1024) -> (I1024, I1024) {
    const H: u32 = 20;
    let a = r.shr_bits(H);
    let a2 = mul(a, a);
    // sin a = a - a^3/3! + ..., cos a = 1 - a^2/2! + ...
    let (mut s, mut c) = (a, one());
    let (mut ts, mut tc) = (a, one());
    let mut k: u64 = 1;
    loop {
        ts = -div_small(mul(ts, a2), (2 * k) * (2 * k + 1));
        tc = -div_small(mul(tc, a2), (2 * k - 1) * (2 * k));
        if ts == I1024::from_i128(0) && tc == I1024::from_i128(0) { break; }
        s = s + ts;
        c = c + tc;
        k += 1;
    }
    for _ in 0..H {
        let s2 = mul(s, c).shl_bits(1);
        let c2 = mul(c, c) - mul(s, s);
        s = s2;
        c = c2;
    }
    (s, c)
}

/// `(sin x, cos x)` of a raw at `f`, at P fraction bits.
pub fn sincos_of_raw<S: WideInt>(raw: S, f: u32) -> (I1024, I1024) {
    let (r, quad) = reduce_half_pi(raw, f);
    let (s, c) = sincos_reduced(r);
    match quad { 0 => (s, c), 1 => (c, -s), 2 => (-s, -c), _ => (-c, s) }
}

/// `sin` / `cos` of a raw at `f`, rounded once.
pub fn sin_cos_to_raw<S: WideInt>(raw: S, f: u32, cos: bool) -> Option<S> {
    let (s, c) = sincos_of_raw(raw, f);
    round_signed::<S>(if cos { c } else { s }, f)
}

/// `a / b` at P fraction bits (I2048 division; the sure path only).
pub(crate) fn div(a: I1024, b: I1024) -> I1024 {
    let num = I2048::from_i1024(a).shl_bits(P);
    let (q, _) = crate::fixed_point::domains::binary_fixed::i2048::i2048_divmod(num, I2048::from_i1024(b));
    q.as_i1024()
}

/// `tan` of a raw at `f`, rounded once (`None` where it leaves the word or
/// cos is zero at P bits).
pub fn tan_to_raw<S: WideInt>(raw: S, f: u32) -> Option<S> {
    let (s, c) = sincos_of_raw(raw, f);
    if c == I1024::from_i128(0) { return None; }
    // |tan| beyond the word: no need to divide
    let (ms, mc) = (if s.is_negative() { -s } else { s }, if c.is_negative() { -c } else { c });
    if ms.bit_length() >= mc.bit_length() + (S::BITS - f) + 2 { return None; }
    // the quotient at P bits can exceed I1024 (|tan| up to 2^255 at
    // Q256.256): divide and round on I2048
    let num = I2048::from_i1024(s).shl_bits(P);
    let (q, _) = crate::fixed_point::domains::binary_fixed::i2048::i2048_divmod(num, I2048::from_i1024(c));
    let sh = (P - f) as usize;
    let up = q.bit((P - f) - 1);
    let r = (q >> sh) + if up { I2048::from_i128(1) } else { I2048::from_i128(0) };
    if r.magnitude_bit_length() > 1020 { return None; }
    from_i1024::<S>(r.as_i1024())
}

/// `sqrt(v)` at P fraction bits for `v >= 0` at P fraction bits.
pub(crate) fn sqrt_p(v: I1024) -> I1024 {
    let wide = I2048::from_i1024(v).shl_bits(P);
    wide.isqrt().as_i1024()
}

/// `asinh` of a raw at `f`, rounded once: odd, and for |x| >= 1
/// `ln|x| + ln(1 + sqrt(1 + 1/x^2))`, so nothing overflows at Q256.256.
pub fn asinh_to_raw<S: WideInt>(raw: S, f: u32) -> Option<S> {
    let neg = raw.is_negative();
    let large = int_bits_of(raw, f) > 0;
    let v = if !large {
        let x = from_raw(raw, f);
        let ax = if neg { -x } else { x };
        // ln(x + sqrt(x^2 + 1)) with x < 1: no cancellation for x >= 0
        let u = ax + sqrt_p(mul(ax, ax) + one());
        ln_of_raw::<I1024>(u, P)
    } else {
        let mag = if neg { raw.checked_neg_wide()? } else { raw };
        // 1/x^2 from the exact 1/x: x itself leaves I1024 at P bits above 2^20
        let inv = inv_of_raw(mag, f);
        let inv2 = mul(inv, inv);
        let t = one() + sqrt_p(one() + inv2);
        ln_of_raw(mag, f) + ln_of_raw::<I1024>(t, P)
    };
    round_signed::<S>(if neg { -v } else { v }, f)
}

/// `atanh` of a raw at `f` (|x| < 1), rounded once: `(ln(1+x) - ln(1-x)) / 2`.
pub fn atanh_to_raw<S: WideInt>(raw: S, f: u32) -> Option<S> {
    let x = from_raw(raw, f);
    let (a, b) = (one() + x, one() - x);
    if !(a > I1024::from_i128(0)) || !(b > I1024::from_i128(0)) { return None; }
    let v = ln_of_raw::<I1024>(a, P) - ln_of_raw::<I1024>(b, P);
    round_signed::<S>(v.shr_bits(1), f)
}

// ============================================================================
// ln, sqrt and the inverse trigonometric sure paths
// ============================================================================

/// `ln` of a positive raw at `f`, rounded once.
pub fn ln_to_raw<S: WideInt>(raw: S, f: u32) -> Option<S> {
    if !(raw > S::from_i128(0)) { return None; }
    round_signed::<S>(ln_of_raw(raw, f), f)
}

/// `sqrt` of a non-negative raw at `f`, exactly rounded: the nearest of
/// `sqrt(raw * 2^f)` from the integer square root with two extra bits (a tie
/// would need an odd square equal to an even number).
pub fn sqrt_to_raw<S: WideInt>(raw: S, f: u32) -> Option<S> {
    if raw.is_negative() { return None; }
    let r = to_i1024(raw).shl_bits(f + 2).isqrt();
    from_i1024::<S>((r + I1024::from_i128(1)).shr_bits(1))
}

/// pi/2 at P fraction bits.
pub(crate) fn half_pi() -> I1024 { (half_pi_2048() >> (PB - P) as usize).as_i1024() }

/// `atan(t)` at P bits for |t| <= 1: six half-angle steps
/// `t <- t / (1 + sqrt(1 + t^2))` (|t| then below 0.0123), the series, and
/// the factor 2^6 back.
fn atan_small(t: I1024) -> I1024 {
    const H: u32 = 6;
    let mut t = t;
    for _ in 0..H { t = div(t, one() + sqrt_p(one() + mul(t, t))); }
    let t2 = mul(t, t);
    let mut sum = t;
    let mut pw = t;
    let mut k: u64 = 1;
    loop {
        pw = -mul(pw, t2);
        let term = div_small(pw, 2 * k + 1);
        if term == I1024::from_i128(0) { break; }
        sum = sum + term;
        k += 1;
    }
    sum.shl_bits(H)
}

/// `atan2(y, x)` at P bits for values at a common scale (not both zero).
pub(crate) fn atan2_values(y: I1024, x: I1024) -> I1024 {
    let (my, mx) = (if y.is_negative() { -y } else { y }, if x.is_negative() { -x } else { x });
    if !(my > mx) {
        let a = atan_small(div(y, x));
        if !x.is_negative() { a } else if y.is_negative() { a - half_pi().shl_bits(1) } else { a + half_pi().shl_bits(1) }
    } else {
        let a = atan_small(div(x, y));
        if y.is_negative() { -half_pi() - a } else { half_pi() - a }
    }
}

/// Two raws at a common scale with the larger near 2^900 (the ratio is all
/// that matters to atan2).
fn common_scale<S: WideInt>(a: S, b: S) -> (I1024, I1024) {
    let (a, b) = (to_i1024(a), to_i1024(b));
    let ma = if a.is_negative() { -a } else { a };
    let mb = if b.is_negative() { -b } else { b };
    let top = ma.bit_length().max(mb.bit_length());
    let sh = 900 - top;
    (a.shl_bits(sh), b.shl_bits(sh))
}

/// `atan2(y, x)` of raws at `f`, rounded once (`None` for (0, 0)).
pub fn atan2_to_raw<S: WideInt>(y: S, x: S, f: u32) -> Option<S> {
    if y == S::from_i128(0) && x == S::from_i128(0) { return None; }
    let (yv, xv) = common_scale(y, x);
    round_signed::<S>(atan2_values(yv, xv), f)
}

/// `atan(x)` of a raw at `f`, rounded once.
pub fn atan_to_raw<S: WideInt>(raw: S, f: u32) -> Option<S> {
    let one_raw = S::from_i128(1).shl_bits(f);
    atan2_to_raw(raw, one_raw, f)
}

/// `asin` / `acos` of a raw at `f` (|x| <= 1), rounded once.
pub fn asin_acos_to_raw<S: WideInt>(raw: S, f: u32, acos: bool) -> Option<S> {
    let x = from_raw(raw, f);
    let c2 = one() - mul(x, x);
    if c2.is_negative() { return None; }
    let a = if c2 == I1024::from_i128(0) {
        if x.is_negative() { -half_pi() } else { half_pi() }
    } else {
        atan2_values(x, sqrt_p(c2))
    };
    round_signed::<S>(if acos { half_pi() - a } else { a }, f)
}

/// `acosh` of a raw at `f` (x >= 1), rounded once:
/// `ln(x + sqrt(x^2 - 1))`, or `ln x + ln(1 + sqrt(1 - 1/x^2))` from 2^10.
pub fn acosh_to_raw<S: WideInt>(raw: S, f: u32) -> Option<S> {
    let one_raw = S::from_i128(1).shl_bits(f);
    if raw < one_raw { return None; }
    // x^2 at P bits fits I1024 only for x below about 2^11
    let v = if int_bits_of(raw, f) <= 10 {
        let x = from_raw(raw, f);
        ln_of_raw::<I1024>(x + sqrt_p(mul(x, x) - one()), P)
    } else {
        let inv = inv_of_raw(raw, f);
        ln_of_raw(raw, f) + ln_of_raw::<I1024>(one() + sqrt_p(one() - mul(inv, inv)), P)
    };
    round_signed::<S>(v, f)
}

/// `tanh` of a raw at `f`, rounded once: `(1 - e^-2|x|) / (1 + e^-2|x|)`
/// with the sign of x (e^-2|x| <= 1, so nothing overflows).
pub fn tanh_to_raw<S: WideInt>(raw: S, f: u32) -> Option<S> {
    let neg = raw.is_negative();
    let one_raw = S::from_i128(1).shl_bits(f);
    if int_bits_of(raw, f) > 9 {
        // |x| >= 512: 1 - tanh|x| < 2^-1400, below half a unit of any format
        return Some(if neg { -one_raw } else { one_raw });
    }
    let x = from_raw(raw, f);
    let ax = if neg { -x } else { x };
    let (m, n) = exp_parts(-ax.shl_bits(1));
    let e = if n >= 0 { m.shl_bits(n as u32) } else { m.shr_bits((-n) as u32) };
    let t = div(one() - e, one() + e);
    round_signed::<S>(if neg { -t } else { t }, f)
}

fn to_i2048<S: WideInt>(raw: S) -> I2048 { I2048::from_i1024(to_i1024(raw)) }

#[cfg(test)]
mod tests {
    use super::*;

    /// ln 2 against its first 128 bits (0xB17217F7D1CF79ABC9E3B39803F2F6AF).
    #[test]
    fn ln2_prefix() {
        let top = ln2().shr_bits(P - 128);
        assert_eq!(i128_of(top) as u128, 0xB17217F7D1CF79ABC9E3B39803F2F6AF);
    }

    /// exp(1) at Q64.64 and exp(10) at Q64.64 against mpmath's nearest raws.
    #[test]
    fn exp_pins() {
        assert_eq!(exp_to_raw::<i128>(one(), 64), Some(50143449209799256683));
        assert_eq!(exp_to_raw::<i128>(mul_small(one(), 10), 64), Some(406316577365116946489258));
    }

    /// ln(e^k) == k and pow(2, 10) == 1024 exactly.
    #[test]
    fn ln_and_pow_pins() {
        let two_raw: i128 = 2 << 64;
        let ten_raw: i128 = 10 << 64;
        assert_eq!(pow_to_raw::<i128>(two_raw, ten_raw, 64), Some(1024i128 << 64));
        assert_eq!(round_signed::<i128>(ln_of_raw::<i128>(1 << 64, 64), 64), Some(0));
    }

    /// pi/2 against its first 128 bits, and sin/cos/tan pins at Q64.64.
    #[test]
    fn trig_pins() {
        let top = (half_pi_2048() >> (PB - 126) as usize).as_i1024();
        assert_eq!(i128_of(top) as u128, 0x6487ed5110b4611a62633145c06e0e68);
        // sin(1), cos(1) at Q64.64 (mpmath nearest raws)
        assert_eq!(sin_cos_to_raw::<i128>(1i128 << 64, 64, false), Some(15522399902203605025));
        assert_eq!(sin_cos_to_raw::<i128>(1i128 << 64, 64, true), Some(9966818358784711826));
        assert_eq!(sin_cos_to_raw::<i128>(-(3i128 << 64), 64, false), Some(-2603204672360199838));
    }

    /// asinh and ln of a value above 2 at P bits (the normalisation shifts right).
    #[test]
    fn asinh_and_wide_ln_pins() {
        assert_eq!(asinh_to_raw::<i32>(3 << 16, 16), Some(119174));
        assert_eq!(asinh_to_raw::<i32>(1 << 15, 16), Some(31537));
        // x = 4089.86: x^2 at P bits would leave I1024
        assert_eq!(asinh_to_raw::<i32>(268031926, 16), Some(590441));
        // ln(24/1024) at Q22.10 sits 6.7e-6 above a tie: -3843.50000666
        assert_eq!(ln_to_raw::<i32>(24, 10), Some(-3844));
        // atan2 at Q22.10: 1608.4997
        assert_eq!(atan2_to_raw::<i32>(1683244501, -7029, 10), Some(1608));
        assert_eq!(sqrt_to_raw::<i32>(2 << 16, 16), Some(92682));
        assert_eq!(asin_acos_to_raw::<i32>(1 << 16, 16, false), Some(102944));
        assert_eq!(asin_acos_to_raw::<i32>(0, 16, true), Some(102944));
        // acosh(27668.2) at Q16.16 (x^2 would leave I1024 at P bits)
        assert_eq!(acosh_to_raw::<i32>(1813252860, 16), Some(715730));
        assert_eq!(round_signed::<i32>(ln_of_raw::<I1024>(mul_small(one(), 6), P), 16), Some(117425));
    }
}

