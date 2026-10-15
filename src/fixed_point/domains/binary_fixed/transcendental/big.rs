//! `Big`: binary floating values with a tracked error bound, the backend of
//! the exact canonical tree evaluation (`docs/design/CORRECT_ROUNDING.md`).
//!
//! A value is `m * 2^(e - P)` with the magnitude `m` normalised to
//! `[2^P, 2^(P+1))` (P = 1000, the sure path's precision), a sign, and a
//! bound `|error| <= 2^err` on its distance from the exact real value of the
//! expression it stands for (`None`: exact). Every operation rounds toward
//! zero at P bits and adds its own bound; the functions are the sure path's
//! algorithms, with the propagated error taken from a derivative bound at
//! the argument. The final rounding to a storage raw is made only when the
//! bound keeps the value away from every rounding boundary; otherwise it
//! answers `Undecided` and the caller keeps its own result.
//!
//! Integer arithmetic only.

use super::sure::{self, P, PB};
use crate::fixed_point::domains::binary_fixed::i2048::i2048_divmod;
use crate::fixed_point::width::WideInt;
use crate::fixed_point::{I1024, I2048};

/// Why a value could not be formed or rounded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BigErr {
    /// The exact value is outside the function's domain.
    Domain,
    /// The error bound reaches a domain edge or a rounding boundary.
    Undecided,
    /// The value is beyond the target or the working range.
    Overflow,
    /// The expression holds something this backend does not evaluate.
    Unsupported,
}

/// A bound `m * 2^x` (m below 2^33), rounded up at every step, so that
/// sums of many bounds grow by their sum, not by a factor two each.
#[derive(Clone, Copy, Debug)]
pub struct Bound { m: u64, x: i64 }

/// An error bound; `None` is exact.
type E = Option<Bound>;

fn bnorm(mut m: u128, mut x: i64) -> Bound {
    while m >= 1u128 << 32 { m = (m + 1) >> 1; x += 1; }
    Bound { m: m as u64, x }
}

/// The bound `2^x`.
fn p2(x: i64) -> E { Some(Bound { m: 1, x }) }

/// `a + b`.
fn sum(a: E, b: E) -> E {
    match (a, b) {
        (None, v) | (v, None) => v,
        (Some(a), Some(b)) => {
            let (hi, lo) = if a.x >= b.x { (a, b) } else { (b, a) };
            let sh = hi.x - lo.x;
            let lo_part: u128 = if sh >= 100 { 1 } else { ((lo.m as u128) + (1u128 << sh) - 1) >> sh };
            Some(bnorm(hi.m as u128 + lo_part, hi.x))
        }
    }
}

fn sums(v: &[E]) -> E { v.iter().fold(None, |acc, x| sum(acc, *x)) }

/// `a * 2^k`.
fn scale(a: E, k: i64) -> E { a.map(|b| Bound { m: b.m, x: b.x + k }) }

/// `a * b`.
fn emul(a: E, b: E) -> E {
    match (a, b) { (Some(a), Some(b)) => Some(bnorm(a.m as u128 * b.m as u128, a.x + b.x)), _ => None }
}

/// `a / (m 2^x)` rounded up (m > 0).
fn ediv(a: E, m: u64, x: i64) -> E {
    a.map(|b| {
        let q = (((b.m as u128) << 64) + m as u128 - 1) / m as u128;
        bnorm(q, b.x - x - 64)
    })
}

/// An exponent `t` with `bound < 2^t` (`None`: exact).
fn top(a: E) -> Option<i64> { a.map(|b| b.x + (64 - b.m.leading_zeros()) as i64) }

/// A tracked binary floating value.
#[derive(Clone, Copy, Debug)]
pub struct Big {
    neg: bool,
    m: I1024,
    e: i64,
    err: E,
}

/// Result of a tracked operation.
pub type BigResult = Result<Big, BigErr>;

const PI64: i64 = P as i64;


fn i2048_abs(v: I2048) -> (bool, I2048) { if v.is_negative() { (true, -v) } else { (false, v) } }

impl Big {
    /// Exact zero.
    pub fn zero() -> Big { Big { neg: false, m: I1024::ZERO, e: 0, err: None } }

    pub fn is_zero(&self) -> bool { self.m == I1024::ZERO }

    /// Sign, magnitude (top bit at P), exponent and error exponent (tests).
    #[cfg(test)]
    pub(crate) fn parts(&self) -> (bool, I1024, i64, Option<i64>) { (self.neg, self.m, self.e, self.err()) }

    pub fn is_exact(&self) -> bool { self.err.is_none() }

    pub fn is_negative(&self) -> bool { self.neg && !self.is_zero() }

    /// An exponent above the error bound (`None`: exact).
    pub fn err(&self) -> Option<i64> { top(self.err) }

    /// `|v|` lies in `[2^e, 2^(e+1))` (nonzero values).
    pub fn exponent(&self) -> i64 { self.e }

    fn with_err(mut self, err: E) -> Big { self.err = err; self }

    /// The sign of the exact value is known (it is exactly zero, or the
    /// bound is below the magnitude).
    pub fn sign_certain(&self) -> bool {
        if self.is_zero() { return self.err.is_none(); }
        match top(self.err) { None => true, Some(x) => x <= self.e }
    }

    /// `mag * 2^scale` with sign, rounded toward zero at P bits: the value
    /// and the bound of that rounding.
    fn norm(neg: bool, mag: I2048, sc: i64) -> (Big, E) {
        if mag == I2048::ZERO { return (Big::zero(), None); }
        let l = mag.bit_length() as i64;
        let e = sc + l - 1;
        let sh = l - 1 - PI64;
        let (m, inexact) = if sh > 0 {
            (mag.shr_bits(sh as u32), mag.low_bits_any(sh as u32))
        } else {
            (mag.shl_bits((-sh) as u32), false)
        };
        (Big { neg, m: m.as_i1024(), e, err: None }, if inexact { p2(e - PI64) } else { None })
    }

    /// A signed I2048 at `frac` fraction bits, plus an engine bound.
    fn from_wide(v: I2048, frac: u32, engine: E) -> Big {
        let (neg, mag) = i2048_abs(v);
        let (b, r) = Big::norm(neg, mag, -(frac as i64));
        b.with_err(sum(r, engine))
    }

    /// A signed I1024 at P fraction bits (a sure-path value), plus an engine bound.
    fn from_fixed(v: I1024, engine: E) -> Big { Big::from_wide(I2048::from_i1024(v), P, engine) }

    /// The exact value `raw * 2^-f` of a word (up to 1024 bits).
    pub fn from_raw<S: WideInt>(raw: S, f: u32) -> Big {
        let v = sure::to_i1024(raw);
        let b = Big::from_wide(I2048::from_i1024(v), f, None);
        debug_assert!(b.err.is_none());
        b
    }

    /// The exact integer `k`.
    pub fn from_int(k: i128) -> Big { Big::from_raw(k, 0) }

    /// pi (from the sure path's 1500-bit pi/2).
    pub fn pi() -> Big { Big::from_wide(sure::half_pi_2048(), PB - 1, p2(-(PB as i64) + 3)) }

    /// Euler's constant gamma, by Brent and McMillan:
    /// `gamma = A/B - ln n` with `B = sum (n^k/k!)^2`, `A = sum (n^k/k!)^2 H_k`,
    /// method error below `pi e^-4n` (n = 180: below 2^-1036). Computed once
    /// per thread.
    pub fn euler_gamma() -> BigResult {
        thread_local! { static G: BigResult = Big::euler_gamma_uncached(); }
        G.with(|g| *g)
    }

    fn euler_gamma_uncached() -> BigResult {
        const N: i128 = 180;
        let n2 = Big::from_int(N * N);
        let (mut b, mut h) = (Big::from_int(1), Big::zero());
        let (mut a_sum, mut b_sum) = (Big::zero(), Big::from_int(1));
        let mut k: i128 = 1;
        loop {
            b = b.mul(n2).div(Big::from_int(k * k))?;
            h = h.add(Big::from_int(1).div(Big::from_int(k))?);
            let a = b.mul(h);
            a_sum = a_sum.add(a);
            b_sum = b_sum.add(b);
            if k > 2 * N && a.e < a_sum.e - PI64 - 8 {
                // past k = 2n each term is below a quarter of the one before
                // (H grows by less than a third), so the tails are below
                // twice the last terms
                a_sum = a_sum.with_err(sum(a_sum.err, p2(a.e + 2)));
                b_sum = b_sum.with_err(sum(b_sum.err, p2(b.e + 2)));
                break;
            }
            k += 1;
        }
        let g = a_sum.div(b_sum)?.sub(Big::from_int(N).ln()?);
        Ok(g.with_err(sum(g.err, p2(-1036))))
    }

    /// The value at P fraction bits as an I1024 (for the sure-path
    /// functions): `None` when the integer part exceeds 2^20.
    fn to_fixed(&self) -> Option<(I1024, E)> {
        if self.is_zero() { return Some((I1024::ZERO, None)); }
        if self.e > 20 { return None; }
        let (v, r) = if self.e >= 0 {
            (self.m.shl_bits(self.e as u32), None)
        } else {
            let s = (-self.e) as u32;
            if s > 1020 { (I1024::ZERO, p2(-PI64)) } else { (self.m.shr_bits(s), if self.m.low_bits_any(s) { p2(-PI64) } else { None }) }
        };
        Some((if self.neg { -v } else { v }, r))
    }

    /// The top 32 bits of the magnitude: `|v|` lies in `[t, t + 1) 2^(e - 31)`.
    fn top32(&self) -> u64 { self.m.shr_bits(P - 31).words[0] }

    /// A bound of `|computed value|` (`None`: zero).
    fn vb(&self) -> E { if self.is_zero() { None } else { Some(bnorm(self.top32() as u128 + 1, self.e - 31)) } }

    pub fn neg(self) -> Big { Big { neg: !self.neg, ..self } }

    pub fn abs(self) -> Big { Big { neg: false, ..self } }

    pub fn add(self, o: Big) -> Big {
        let err = sum(self.err, o.err);
        if o.is_zero() { return self.with_err(err); }
        if self.is_zero() { return o.with_err(err); }
        let (a, b) = if self.e >= o.e { (self, o) } else { (o, self) };
        let d = a.e - b.e;
        if d > PI64 + 8 {
            // b lies below a's last place: count it as error
            return a.with_err(sum(err, p2(b.e + 1)));
        }
        let am = I2048::from_i1024(a.m).shl_bits(d as u32);
        let bm = I2048::from_i1024(b.m);
        let (neg, mag) = if a.neg == b.neg {
            (a.neg, am + bm)
        } else if !(am < bm) {
            (a.neg, am - bm)
        } else {
            (b.neg, bm - am)
        };
        let (r, round) = Big::norm(neg, mag, b.e - PI64);
        r.with_err(sum(err, round))
    }

    pub fn sub(self, o: Big) -> Big { self.add(o.neg()) }

    pub fn mul(self, o: Big) -> Big {
        let err = sums(&[emul(self.vb(), o.err), emul(o.vb(), self.err), emul(self.err, o.err)]);
        if self.is_zero() || o.is_zero() { return Big::zero().with_err(err); }
        let prod = self.m.mul_to_i2048(o.m);
        let (r, round) = Big::norm(self.neg != o.neg, prod, self.e + o.e - 2 * PI64);
        r.with_err(sum(err, round))
    }

    pub fn div(self, o: Big) -> BigResult {
        if o.is_zero() { return Err(if o.err.is_none() { BigErr::Domain } else { BigErr::Undecided }); }
        // the divisor must stay away from zero: |b'| >= |b| / 2
        if let Some(eb) = top(o.err) { if eb > o.e - 1 { return Err(BigErr::Undecided); } }
        let num = I2048::from_i1024(self.m).shl_bits(P + 1);
        let (q, rem) = if self.is_zero() { (I2048::ZERO, I2048::ZERO) } else { i2048_divmod(num, I2048::from_i1024(o.m)) };
        let (r, mut round) = Big::norm(self.neg != o.neg, q, self.e - o.e - PI64 - 1);
        if rem != I2048::ZERO && round.is_none() && !r.is_zero() { round = p2(r.e - PI64); }
        // |a'/b' - a/b| <= (|da| + |a/b| |db|) / |b'|; |b'| >= |b| - |db|,
        // taken as (t - 1) 2^(e - 31) when |db| is below 2^(e - 32), else 2^(e - 1)
        let num = sum(self.err, emul(r.vb(), o.err));
        let tight = match top(o.err) { None => true, Some(t) => t <= o.e - 32 };
        let prop = if tight { ediv(num, o.top32() - 1, o.e - 31) } else { scale(num, -(o.e - 1)) };
        Ok(r.with_err(sum(prop, round)))
    }

    /// Halving and doubling are exact.
    pub fn shl(self, k: i64) -> Big { if self.is_zero() { self.with_err(scale(self.err, k)) } else { Big { e: self.e + k, err: scale(self.err, k), ..self } } }

    pub fn sqrt(self) -> BigResult {
        if self.is_zero() {
            return Ok(match top(self.err) { None => Big::zero(), Some(x) => Big::zero().with_err(p2(x.div_euclid(2) + 1)) });
        }
        if !self.sign_certain() { return Err(BigErr::Undecided); }
        if self.neg { return Err(BigErr::Domain); }
        // m * 2^(e - P) with e - P made even
        let (mm, ee) = if (self.e - PI64).rem_euclid(2) == 0 { (I2048::from_i1024(self.m), self.e - PI64) } else { (I2048::from_i1024(self.m).shl_bits(1), self.e - PI64 - 1) };
        let wide = mm.shl_bits(P + 2);
        let root = wide.isqrt();
        let exact_root = root * root == wide;
        let (r, mut round) = Big::norm(false, root, (ee - (PI64 + 2)) / 2);
        if !exact_root && round.is_none() { round = p2(r.e - PI64); }
        // |sqrt(x + d) - sqrt(x)| <= |d| / sqrt(x), sqrt(x) >= 2^floor(e/2)
        let prop = scale(self.err, -self.e.div_euclid(2));
        Ok(r.with_err(sum(prop, round)))
    }

    pub fn exp(self) -> BigResult {
        if self.is_zero() && self.err.is_none() { return Ok(Big::from_int(1)); }
        if let Some(x) = top(self.err) { if x > -1 { return Err(BigErr::Undecided); } }
        let (xf, conv) = match self.to_fixed() {
            Some(v) => v,
            None => {
                // |x| > 2^20: overflow, or below every format
                if self.neg { return Ok(Big::zero().with_err(p2(-(1 << 20)))); }
                return Err(BigErr::Overflow);
            }
        };
        let (m, n) = sure::exp_parts(xf);
        let (r, round) = Big::norm(false, I2048::from_i1024(m), n - PI64);
        // relative engine error 2^(48 - P); |e^d - 1| <= 2|d| for |d| <= 1/2
        let d = sum(self.err, conv);
        let prop = scale(d, r.e + 2);
        Ok(r.with_err(sums(&[prop, round, p2(r.e + 1 - PI64 + 48)])))
    }

    pub fn ln(self) -> BigResult {
        if self.is_zero() { return Err(if self.err.is_none() { BigErr::Domain } else { BigErr::Undecided }); }
        if !self.sign_certain() { return Err(BigErr::Undecided); }
        if self.neg { return Err(BigErr::Domain); }
        if self.err.is_none() && self.e == 0 && self.m == sure::one() { return Ok(Big::zero()); }
        if let Some(x) = top(self.err) { if x > self.e - 1 { return Err(BigErr::Undecided); } }
        let lm = sure::ln_of_raw::<I1024>(self.m, P);
        let v = lm + sure::mul_small(sure::ln2(), self.e);
        let bits_e = 64 - self.e.unsigned_abs().leading_zeros() as i64;
        let engine = p2(-PI64 + 52 + bits_e);
        let prop = scale(self.err, 1 - self.e);
        Ok(Big::from_fixed(v, sum(engine, prop)))
    }

    /// `(sin x, cos x)`.
    pub fn sin_cos(self) -> Result<(Big, Big), BigErr> {
        if self.is_zero() && self.err.is_none() { return Ok((Big::zero(), Big::from_int(1))); }
        if let Some(x) = top(self.err) { if x > 0 { return Err(BigErr::Undecided); } }
        // x at PB fraction bits on I2048
        let (x, conv) = if self.is_zero() {
            (I2048::ZERO, None)
        } else {
            let k = self.e + PB as i64 - PI64;
            if k > 1000 { return Err(BigErr::Unsupported); }
            let m = I2048::from_i1024(self.m);
            let (v, c) = if k >= 0 { (m.shl_bits(k as u32), None) } else if -k > 1020 { (I2048::ZERO, p2(-(PB as i64))) } else { (m.shr_bits((-k) as u32), if m.low_bits_any((-k) as u32) { p2(-(PB as i64)) } else { None }) };
            (if self.neg { -v } else { v }, c)
        };
        let (r, quad) = sure::reduce_half_pi_wide(x);
        let (s, c) = sure::sincos_reduced(r);
        let (s, c) = match quad { 0 => (s, c), 1 => (c, -s), 2 => (-s, -c), _ => (-c, s) };
        let reduction = if self.is_zero() { None } else { p2(self.e + 3 - PB as i64) };
        let engine = sums(&[p2(-PI64 + 26), reduction, conv]);
        let err = sum(engine, self.err);
        Ok((Big::from_fixed(s, err), Big::from_fixed(c, err)))
    }

    /// `atan2(self, x)`, `self` the ordinate.
    pub fn atan2(self, x: Big) -> BigResult {
        let y = self;
        if y.is_zero() && x.is_zero() {
            return Err(if y.err.is_none() && x.err.is_none() { BigErr::Domain } else { BigErr::Undecided });
        }
        let top_e = if y.is_zero() { x.e } else if x.is_zero() { y.e } else { y.e.max(x.e) };
        // the point must stay away from the origin: r >= 2^top
        let d = sum(y.err, x.err);
        if let Some(dv) = top(d) { if dv > top_e - 2 { return Err(BigErr::Undecided); } }
        let at_top = |v: Big| -> I1024 {
            if v.is_zero() { return I1024::ZERO; }
            let s = (top_e - v.e) as u32;
            let m = if s > 1020 { I1024::ZERO } else { v.m.shr_bits(s) };
            if v.neg { -m } else { m }
        };
        let a = sure::atan2_values(at_top(y), at_top(x));
        let engine = p2(-PI64 + 20);
        let prop = scale(d, 1 - top_e);
        Ok(Big::from_fixed(a, sum(engine, prop)))
    }

    pub fn atan(self) -> BigResult { self.atan2(Big::from_int(1)) }

    pub fn tan(self) -> BigResult { let (s, c) = self.sin_cos()?; s.div(c) }

    /// `1 - x^2` as `(1 - |x|)(1 + |x|)` (no cancellation near |x| = 1).
    fn one_minus_sq(self) -> Big {
        let one = Big::from_int(1);
        let ax = self.abs();
        one.sub(ax).mul(one.add(ax))
    }

    pub fn asin(self) -> BigResult {
        let g = self.one_minus_sq();
        if !g.sign_certain() { return Err(BigErr::Undecided); }
        if g.is_negative() { return Err(BigErr::Domain); }
        self.atan2(g.sqrt()?)
    }

    pub fn acos(self) -> BigResult {
        let g = self.one_minus_sq();
        if !g.sign_certain() { return Err(BigErr::Undecided); }
        if g.is_negative() { return Err(BigErr::Domain); }
        g.sqrt()?.atan2(self)
    }

    /// `sinh x` for |x| < 1/8 by its series (no cancellation).
    fn sinh_series(self) -> Big {
        let x2 = self.mul(self);
        let mut term = self;
        let mut sum = self;
        let mut k: i128 = 1;
        loop {
            term = match term.mul(x2).div(Big::from_int((2 * k) * (2 * k + 1))) { Ok(t) => t, Err(_) => break };
            if term.is_zero() || term.e < sum.e - PI64 - 8 {
                // the rest is below twice the last term
                let tail = if term.is_zero() { None } else { p2(term.e + 2) };
                return sum.add(term).with_err(sums(&[sum.add(term).err, tail]));
            }
            sum = sum.add(term);
            k += 1;
        }
        sum
    }

    /// `(e^|x|, e^-|x|)`.
    fn exp_pair(self) -> Result<(Big, Big), BigErr> {
        let a = self.abs().exp()?;
        let b = Big::from_int(1).div(a)?;
        Ok((a, b))
    }

    pub fn sinh(self) -> BigResult {
        if self.is_zero() && self.err.is_none() { return Ok(Big::zero()); }
        if !self.is_zero() && self.e < -3 { return Ok(self.sinh_series()); }
        let (a, b) = self.exp_pair()?;
        let v = a.sub(b).shl(-1);
        Ok(if self.neg { v.neg() } else { v })
    }

    pub fn cosh(self) -> BigResult {
        if self.is_zero() && self.err.is_none() { return Ok(Big::from_int(1)); }
        let (a, b) = self.exp_pair()?;
        Ok(a.add(b).shl(-1))
    }

    pub fn tanh(self) -> BigResult {
        if self.is_zero() && self.err.is_none() { return Ok(Big::zero()); }
        if !self.is_zero() && self.e >= 10 {
            // 1 - tanh|x| < 2 e^-2|x| < 2^-1400
            let one = Big::from_int(1).with_err(p2(-1400));
            return Ok(if self.neg { one.neg() } else { one });
        }
        let v = if !self.is_zero() && self.e < -3 {
            let s = self.abs().sinh_series();
            let c = Big::from_int(1).add(s.mul(s)).sqrt()?;
            s.div(c)?
        } else {
            let t = self.abs().shl(1).neg().exp()?;
            let one = Big::from_int(1);
            one.sub(t).div(one.add(t))?
        };
        Ok(if self.neg { v.neg() } else { v })
    }

    pub fn asinh(self) -> BigResult {
        let ax = self.abs();
        let u = ax.add(ax.mul(ax).add(Big::from_int(1)).sqrt()?);
        let v = u.ln()?;
        Ok(if self.neg { v.neg() } else { v })
    }

    pub fn acosh(self) -> BigResult {
        let one = Big::from_int(1);
        let g = self.sub(one);
        if !g.sign_certain() { return Err(BigErr::Undecided); }
        if g.is_negative() { return Err(BigErr::Domain); }
        let r = g.mul(self.add(one)).sqrt()?;
        self.add(r).ln()
    }

    pub fn atanh(self) -> BigResult {
        let one = Big::from_int(1);
        let g = one.sub(self.abs());
        if !g.sign_certain() || g.is_zero() { return Err(if g.is_zero() && g.err.is_none() { BigErr::Domain } else { BigErr::Undecided }); }
        if g.is_negative() { return Err(BigErr::Domain); }
        let v = one.add(self).ln()?.sub(one.sub(self).ln()?).shl(-1);
        Ok(v)
    }

    /// `self^y`: `exp(y ln x)` for x > 0; 0 for an exact zero base and y > 0.
    pub fn pow(self, y: Big) -> BigResult {
        if y.is_zero() && y.err.is_none() { return Ok(Big::from_int(1)); }
        if self.is_zero() && self.err.is_none() {
            if !y.sign_certain() { return Err(BigErr::Undecided); }
            return if y.is_negative() || y.is_zero() { Err(BigErr::Domain) } else { Ok(Big::zero()) };
        }
        if !self.sign_certain() { return Err(BigErr::Undecided); }
        if self.is_negative() { return Err(BigErr::Unsupported); }
        y.mul(self.ln()?).exp()
    }

    /// The nearest raw at `f` fraction bits, ties toward +infinity, when
    /// the bound decides it.
    pub fn round_to_raw<S: WideInt>(&self, f: u32) -> Result<S, BigErr> {
        let fi = f as i64;
        let tiny = |e_hi: i64| -> Result<S, BigErr> {
            // |v 2^f| <= 2^e_hi: zero when below 1/4 and the error too
            match top(self.err) { Some(x) if x + fi > -2 => Err(BigErr::Undecided), _ if e_hi < -2 => Ok(S::ZERO), _ => Err(BigErr::Undecided) }
        };
        if self.is_zero() {
            return match self.err { None => Ok(S::ZERO), Some(_) => tiny(i64::MIN / 4) };
        }
        let s = PI64 - fi - self.e;
        if s > 1000 { return tiny(self.e + 1 + fi); }
        let mag = if s <= 0 {
            // an integer at f bits: the boundaries are half a unit away
            if let Some(x) = top(self.err) { if x + fi > -1 { return Err(BigErr::Undecided); } }
            let sh = (-s) as u32;
            if self.m.bit_length() + sh > S::BITS { return Err(BigErr::Overflow); }
            I2048::from_i1024(self.m).shl_bits(sh)
        } else {
            let sh = s as u32;
            let q = I2048::from_i1024(self.m.shr_bits(sh));
            let rem = self.m - self.m.shr_bits(sh).shl_bits(sh);
            let half = I1024::from_i128(1).shl_bits(sh - 1);
            let diff = rem - half;
            let up = if self.neg { diff > I1024::ZERO } else { !diff.is_negative() };
            if let Some(x) = top(self.err) {
                // decided when |diff| 2^-s > error 2^f, error < 2^x
                let t = x + fi + s;
                let dm = if diff.is_negative() { -diff } else { diff };
                let b = dm.bit_length() as i64;
                if b == 0 || b - 1 < t { return Err(BigErr::Undecided); }
            }
            if up { q + I2048::from_i128(1) } else { q }
        };
        // into the word, -2^(BITS-1) included
        let n = mag.bit_length();
        let fits = n < S::BITS || (self.neg && n == S::BITS && mag == I2048::from_i128(1).shl_bits(S::BITS - 1));
        if !fits { return Err(BigErr::Overflow); }
        let mut out = S::ZERO;
        let mut i = 0;
        while i < n {
            if mag.bit(i) { out = out.checked_add_wide(S::from_i128(1).shl_bits(i)).unwrap_or(out); }
            i += 1;
        }
        if self.neg {
            if n == S::BITS { return Ok(S::from_i128(1).shl_bits(S::BITS - 1)); }
            out = out.checked_neg_wide().ok_or(BigErr::Overflow)?;
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q64(b: BigResult) -> i128 { b.expect("value").round_to_raw::<i128>(64).expect("decided") }

    #[test]
    fn pins_against_the_sure_path() {
        let one = Big::from_int(1);
        assert_eq!(q64(one.exp()), 50143449209799256683);
        assert_eq!(q64(Big::from_int(10).exp()), 406316577365116946489258);
        assert_eq!(q64(one.sin_cos().map(|p| p.0)), 15522399902203605025);
        assert_eq!(q64(one.sin_cos().map(|p| p.1)), 9966818358784711826);
        assert_eq!(q64(Big::from_int(-3).sin_cos().map(|p| p.0)), -2603204672360199838);
        assert_eq!(Big::from_int(2).sqrt().unwrap().round_to_raw::<i32>(16).unwrap(), 92682);
        assert_eq!(Big::from_int(1).asin().unwrap().round_to_raw::<i32>(16).unwrap(), 102944);
        assert_eq!(Big::from_int(0).acos().unwrap().round_to_raw::<i32>(16).unwrap(), 102944);
        assert_eq!(Big::from_int(3).asinh().unwrap().round_to_raw::<i32>(16).unwrap(), 119174);
        assert_eq!(Big::from_int(6).ln().unwrap().round_to_raw::<i32>(16).unwrap(), 117425);
        // gamma = 0.57721566490153286060651209008240243104215933593992...
        // nearest at Q64.64: floor(gamma 2^64 + 1/2) (mpmath)
        assert_eq!(q64(Big::euler_gamma()), 10647749645774669733);
        let ge = Big::euler_gamma().unwrap().err().unwrap();
        assert!(ge < -900, "gamma bound 2^{ge}");
    }

    #[test]
    fn exact_operations_stay_exact_and_ties_go_up() {
        let a = Big::from_raw(3i32, 1); // 1.5
        let b = Big::from_raw(-5i32, 2); // -1.25
        let p = a.mul(b); // -1.875
        assert!(p.is_exact());
        assert_eq!(p.round_to_raw::<i32>(2).unwrap(), -7); // -1.75 or -2.0: -1.875 is a tie at 2 bits... -7.5 -> -7
        assert_eq!(Big::from_raw(15i32, 3).round_to_raw::<i32>(2).unwrap(), 8); // 7.5 -> 8
        let q = Big::from_int(1).div(Big::from_int(3)).unwrap();
        assert!(!q.is_exact());
        assert_eq!(q.round_to_raw::<i32>(16).unwrap(), 21845);
        assert_eq!(Big::from_int(4).sqrt().unwrap().round_to_raw::<i32>(0).unwrap(), 2);
        assert!(Big::from_int(4).sqrt().unwrap().is_exact());
    }

    #[test]
    fn domains_and_undecided() {
        assert_eq!(Big::from_int(-1).ln().unwrap_err(), BigErr::Domain);
        assert_eq!(Big::from_int(2).asin().unwrap_err(), BigErr::Domain);
        assert_eq!(Big::from_int(1).atanh().unwrap_err(), BigErr::Domain);
        assert_eq!(Big::from_int(1).div(Big::zero()).unwrap_err(), BigErr::Domain);
        // a value whose sign is unknown cannot take ln
        let z = Big::zero().with_err(p2(-900));
        assert_eq!(z.ln().unwrap_err(), BigErr::Undecided);
    }
}
