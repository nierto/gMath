//! `Mpf<C>`: binary floating values at a chosen precision with a tracked
//! error bound, the backend of the exact canonical tree evaluation
//! (`docs/design/CORRECT_ROUNDING.md`).
//!
//! The precision is a type: `C` names a capacity of `C::L` 64-bit limbs
//! (192 bits for `C3` up to 4160 bits for `C65`), every value and every
//! temporary lives on the stack, and the evaluator starts at a capacity
//! just above the format's width and moves up only when the bound cannot
//! decide the final rounding.
//!
//! A value is `M * 2^(e - (64L - 1))`, `M` an integer with its top bit at
//! `64L - 1`, a sign, and a bound `|error| <= m 2^x` on its distance from
//! the exact real value it stands for (`None`: exact). Every operation
//! rounds toward zero and adds its own bound; the functions add their
//! method and rounding bounds and the propagated error from a derivative
//! bound at the argument. Integer arithmetic only.

use crate::fixed_point::width::WideInt;
use core::cmp::Ordering;

// ============================================================================
// Capacities
// ============================================================================

/// A precision: `L` limbs, with arrays of `L` and `2L + 2` limbs.
pub trait Cap: Copy + core::fmt::Debug + 'static {
    const L: usize;
    type A: Copy + core::fmt::Debug + AsRef<[u64]> + AsMut<[u64]>;
    type W: Copy + AsRef<[u64]> + AsMut<[u64]>;
    fn a() -> Self::A;
    fn w() -> Self::W;
}

macro_rules! cap {
    ($name:ident, $l:literal) => {
        #[derive(Clone, Copy, Debug)]
        pub struct $name;
        impl Cap for $name {
            const L: usize = $l;
            type A = [u64; $l];
            type W = [u64; 2 * $l + 2];
            #[inline] fn a() -> [u64; $l] { [0; $l] }
            #[inline] fn w() -> [u64; 2 * $l + 2] { [0; 2 * $l + 2] }
        }
    };
}
cap!(C2, 2);
cap!(C3, 3);
cap!(C5, 5);
cap!(C9, 9);
cap!(C17, 17);
cap!(C33, 33);
cap!(C65, 65);

// ============================================================================
// Limb arithmetic (little-endian slices)
// ============================================================================

fn bitlen(a: &[u64]) -> i64 {
    for i in (0..a.len()).rev() {
        if a[i] != 0 { return 64 * i as i64 + 64 - a[i].leading_zeros() as i64; }
    }
    0
}

fn all_zero(a: &[u64]) -> bool { a.iter().all(|&x| x == 0) }

/// Any of the bits below `k`.
fn low_any(a: &[u64], k: i64) -> bool {
    if k <= 0 { return false; }
    let full = ((k / 64) as usize).min(a.len());
    if a[..full].iter().any(|&x| x != 0) { return true; }
    let r = (k % 64) as u32;
    r > 0 && full < a.len() && a[full] & ((1u64 << r) - 1) != 0
}

/// `dst = src >> k`, truncated to `dst`.
fn shr_to(src: &[u64], k: i64, dst: &mut [u64]) {
    let w = (k / 64) as usize;
    let b = (k % 64) as u32;
    for i in 0..dst.len() {
        let lo = src.get(i + w).copied().unwrap_or(0);
        let hi = src.get(i + w + 1).copied().unwrap_or(0);
        dst[i] = if b == 0 { lo } else { (lo >> b) | (hi << (64 - b)) };
    }
}

/// `dst = src << k`, truncated to `dst`.
fn shl_to(src: &[u64], k: i64, dst: &mut [u64]) {
    let w = (k / 64) as usize;
    let b = (k % 64) as u32;
    for i in 0..dst.len() {
        if i < w { dst[i] = 0; continue; }
        let j = i - w;
        let lo = src.get(j).copied().unwrap_or(0);
        let below = if j >= 1 { src.get(j - 1).copied().unwrap_or(0) } else { 0 };
        dst[i] = if b == 0 { lo } else { (lo << b) | (below >> (64 - b)) };
    }
}

/// `a += b` (b no longer than a); the carry out.
fn add_in(a: &mut [u64], b: &[u64]) -> u64 {
    let mut c = 0u64;
    for i in 0..a.len() {
        let bi = b.get(i).copied().unwrap_or(0);
        if bi == 0 && c == 0 && i >= b.len() { break; }
        let (s1, o1) = a[i].overflowing_add(bi);
        let (s2, o2) = s1.overflowing_add(c);
        a[i] = s2;
        c = (o1 as u64) + (o2 as u64);
    }
    c
}

/// `a -= b` (b no longer than a, a >= b).
fn sub_in(a: &mut [u64], b: &[u64]) {
    let mut br = 0u64;
    for i in 0..a.len() {
        let bi = b.get(i).copied().unwrap_or(0);
        if bi == 0 && br == 0 && i >= b.len() { break; }
        let (d1, o1) = a[i].overflowing_sub(bi);
        let (d2, o2) = d1.overflowing_sub(br);
        a[i] = d2;
        br = (o1 as u64) + (o2 as u64);
    }
    debug_assert!(br == 0, "sub_in: negative result");
}

fn cmp_limbs(a: &[u64], b: &[u64]) -> Ordering {
    let n = a.len().max(b.len());
    for i in (0..n).rev() {
        let (x, y) = (a.get(i).copied().unwrap_or(0), b.get(i).copied().unwrap_or(0));
        if x != y { return x.cmp(&y); }
    }
    Ordering::Equal
}

/// `out = a * b` (`out` at least `a.len() + b.len()` limbs).
fn mul_to(a: &[u64], b: &[u64], out: &mut [u64]) {
    for x in out.iter_mut() { *x = 0; }
    for i in 0..a.len() {
        if a[i] == 0 { continue; }
        let mut c: u128 = 0;
        for j in 0..b.len() {
            let t = (a[i] as u128) * (b[j] as u128) + out[i + j] as u128 + c;
            out[i + j] = t as u64;
            c = t >> 64;
        }
        let mut k = i + b.len();
        while c != 0 {
            let t = out[k] as u128 + c;
            out[k] = t as u64;
            c = t >> 64;
            k += 1;
        }
    }
}

/// `a *= k`; the carry limb.
fn mul_small_in(a: &mut [u64], k: u64) -> u64 {
    let mut c: u128 = 0;
    for x in a.iter_mut() {
        let t = (*x as u128) * (k as u128) + c;
        *x = t as u64;
        c = t >> 64;
    }
    c as u64
}

/// `a /= d`; the remainder.
fn div_small_in(a: &mut [u64], d: u64) -> u64 {
    let mut r: u128 = 0;
    for i in (0..a.len()).rev() {
        let cur = (r << 64) | a[i] as u128;
        a[i] = (cur / d as u128) as u64;
        r = cur % d as u128;
    }
    r as u64
}

/// `q = floor(u / v)` (Knuth, algorithm D). `v` has a nonzero top limb,
/// `q` at least `u.len() - v.len() + 1` limbs, `scratch` at least
/// `u.len() + 1` and `v.len()` limbs. Returns whether the remainder is
/// nonzero.
fn divrem(u: &[u64], v: &[u64], q: &mut [u64], un: &mut [u64], vn: &mut [u64]) -> bool {
    let n = v.len();
    let m = u.len() - n;
    for x in q.iter_mut() { *x = 0; }
    if n == 1 {
        un[..u.len()].copy_from_slice(u);
        let r = div_small_in(&mut un[..u.len()], v[0]);
        q[..u.len()].copy_from_slice(&un[..u.len()]);
        return r != 0;
    }
    let s = v[n - 1].leading_zeros();
    shl_to(v, s as i64, &mut vn[..n]);
    shl_to(u, s as i64, &mut un[..u.len() + 1]);
    let b: u128 = 1 << 64;
    for j in (0..=m).rev() {
        let num = ((un[j + n] as u128) << 64) | un[j + n - 1] as u128;
        let mut qhat = num / vn[n - 1] as u128;
        let mut rhat = num % vn[n - 1] as u128;
        while qhat >= b || qhat * vn[n - 2] as u128 > ((rhat << 64) | un[j + n - 2] as u128) {
            qhat -= 1;
            rhat += vn[n - 1] as u128;
            if rhat >= b { break; }
        }
        // un[j..j+n+1] -= qhat * vn
        let mut borrow: i128 = 0;
        let mut carry: u128 = 0;
        for i in 0..n {
            let p = qhat * vn[i] as u128 + carry;
            carry = p >> 64;
            let t = un[i + j] as i128 - borrow - (p as u64) as i128;
            un[i + j] = t as u64;
            borrow = if t < 0 { 1 } else { 0 };
        }
        let t = un[j + n] as i128 - borrow - carry as i128;
        un[j + n] = t as u64;
        if t < 0 {
            qhat -= 1;
            let mut c: u128 = 0;
            for i in 0..n {
                let s2 = un[i + j] as u128 + vn[i] as u128 + c;
                un[i + j] = s2 as u64;
                c = s2 >> 64;
            }
            un[j + n] = un[j + n].wrapping_add(c as u64);
        }
        q[j] = qhat as u64;
    }
    !all_zero(&un[..n])
}

// ============================================================================
// Error bounds
// ============================================================================

/// A bound `m * 2^x` (m below 2^33), rounded up at every step, so that sums
/// of many bounds grow by their sum.
#[derive(Clone, Copy, Debug)]
pub struct Bound { m: u64, x: i64 }

/// An error bound; `None` is exact.
pub type E = Option<Bound>;

fn bnorm(mut m: u128, mut x: i64) -> Bound {
    if m >= 1u128 << 32 {
        // one shift, rounded up; a carry to 2^32 takes one more
        let k = 96 - m.leading_zeros();
        m = (m + (1u128 << k) - 1) >> k;
        x += k as i64;
        if m >= 1u128 << 32 { m = (m + 1) >> 1; x += 1; }
    }
    Bound { m: m as u64, x }
}

/// The bound `2^x`.
pub fn p2(x: i64) -> E { Some(Bound { m: 1, x }) }

/// `a + b`.
pub fn sum(a: E, b: E) -> E {
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

pub fn sums(v: &[E]) -> E { v.iter().fold(None, |acc, x| sum(acc, *x)) }

/// `a * 2^k`.
pub fn scale(a: E, k: i64) -> E { a.map(|b| Bound { m: b.m, x: b.x + k }) }

/// `a * b`.
pub fn emul(a: E, b: E) -> E {
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
pub fn top(a: E) -> Option<i64> { a.map(|b| b.x + (64 - b.m.leading_zeros()) as i64) }

/// Why a value could not be formed or rounded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MpfErr {
    /// The exact value is outside the function's domain.
    Domain,
    /// The error bound reaches a domain edge or a rounding boundary.
    Undecided,
    /// The value lies within the bound of a rounding tie, the bound below a
    /// millionth of a unit.
    NearTie,
    /// The value is beyond the target or the working range.
    Overflow,
    /// The expression holds something this backend does not evaluate.
    Unsupported,
}

pub type MpfResult<C> = Result<Mpf<C>, MpfErr>;

// ============================================================================
// The floating type
// ============================================================================

/// A tracked binary floating value at capacity `C`.
#[derive(Clone, Copy, Debug)]
pub struct Mpf<C: Cap> {
    neg: bool,
    m: C::A,
    e: i64,
    err: E,
}

impl<C: Cap> Mpf<C> {
    /// Bits of the mantissa.
    pub const P: i64 = 64 * C::L as i64;

    pub fn zero() -> Self { Mpf { neg: false, m: C::a(), e: 0, err: None } }

    pub fn is_zero(&self) -> bool { all_zero(self.m.as_ref()) }

    pub fn is_exact(&self) -> bool { self.err.is_none() }

    pub fn is_negative(&self) -> bool { self.neg && !self.is_zero() }

    pub fn err(&self) -> E { self.err }

    /// `|v|` lies in `[2^e, 2^(e+1))` (nonzero values).
    pub fn exponent(&self) -> i64 { self.e }

    pub fn with_err(mut self, err: E) -> Self { self.err = err; self }

    /// One unit of the last place: `2^(e - P + 1)`.
    fn ulp_exp(&self) -> i64 { self.e - Self::P + 1 }

    /// The sign of the exact value is known.
    pub fn sign_certain(&self) -> bool {
        if self.is_zero() { return self.err.is_none(); }
        match top(self.err) { None => true, Some(x) => x <= self.e }
    }

    /// `t * 2^sc` with sign, rounded toward zero to the capacity: the value
    /// and the bound of that rounding.
    fn norm(neg: bool, t: &[u64], sc: i64) -> (Self, E) {
        let b = bitlen(t);
        if b == 0 { return (Self::zero(), None); }
        let e = sc + b - 1;
        let sh = b - Self::P;
        let mut m = C::a();
        let inexact = if sh > 0 {
            shr_to(t, sh, m.as_mut());
            low_any(t, sh)
        } else {
            shl_to(t, -sh, m.as_mut());
            false
        };
        let r = Mpf { neg, m, e, err: None };
        let round = if inexact { p2(r.ulp_exp()) } else { None };
        (r, round)
    }

    /// The exact value `raw * 2^-f` of a word, rounded to the capacity when
    /// the word is wider.
    pub fn from_raw<S: WideInt>(raw: S, f: u32) -> Self {
        let neg = raw.is_negative();
        let mut t = [0u64; 18];
        let words = ((S::BITS + 63) / 64) as usize;
        match if neg { raw.checked_neg_wide() } else { Some(raw) } {
            Some(mag) => for i in 0..words { t[i] = mag.word(i as u32); },
            None => { let k = S::BITS - 1; t[(k / 64) as usize] = 1u64 << (k % 64); }
        }
        if S::BITS < 64 { t[0] &= (1u64 << S::BITS) - 1; }
        let (r, round) = Self::norm(neg, &t[..words + 1], -(f as i64));
        r.with_err(round)
    }

    pub fn from_int(k: i128) -> Self { Self::from_raw(k, 0) }

    /// From limbs of another precision (`top` its top exponent).
    fn from_limbs(neg: bool, limbs: &[u64], e: i64, err: E) -> Self {
        let (r, round) = Self::norm(neg, limbs, e - (64 * limbs.len() as i64 - 1));
        r.with_err(sum(err, round))
    }

    pub fn neg(self) -> Self { Mpf { neg: !self.neg, ..self } }

    pub fn abs(self) -> Self { Mpf { neg: false, ..self } }

    /// `self * 2^k` (exact).
    pub fn shl(self, k: i64) -> Self {
        if self.is_zero() { return self.with_err(scale(self.err, k)); }
        Mpf { e: self.e + k, err: scale(self.err, k), ..self }
    }

    /// The top 32 bits of the magnitude: `|v|` lies in `[t, t + 1) 2^(e - 31)`.
    fn top32(&self) -> u64 { self.m.as_ref()[C::L - 1] >> 32 }

    /// A bound of the computed magnitude (`None`: zero).
    fn vb(&self) -> E { if self.is_zero() { None } else { Some(bnorm(self.top32() as u128 + 1, self.e - 31)) } }

    pub fn add(self, o: Self) -> Self {
        let err = sum(self.err, o.err);
        if o.is_zero() { return self.with_err(err); }
        if self.is_zero() { return o.with_err(err); }
        let (a, b) = if self.e >= o.e { (self, o) } else { (o, self) };
        let d = a.e - b.e;
        if d > Self::P + 2 {
            // b lies wholly below a's last place: count it as error
            return a.with_err(sum(err, p2(b.e + 1)));
        }
        // align: a << d over b, then add, in 2L+2 limbs (d <= P + 2)
        let mut ta = C::w();
        let mut tb = C::w();
        shl_to(a.m.as_ref(), d, ta.as_mut());
        tb.as_mut()[..C::L].copy_from_slice(b.m.as_ref());
        let neg = if a.neg == b.neg {
            add_in(ta.as_mut(), tb.as_ref());
            a.neg
        } else if cmp_limbs(ta.as_ref(), tb.as_ref()) != Ordering::Less {
            sub_in(ta.as_mut(), tb.as_ref());
            a.neg
        } else {
            sub_in(tb.as_mut(), ta.as_ref());
            ta = tb;
            b.neg
        };
        let (r, round) = Self::norm(neg, ta.as_ref(), b.e - Self::P + 1);
        r.with_err(sum(err, round))
    }

    pub fn sub(self, o: Self) -> Self { self.add(o.neg()) }

    pub fn mul(self, o: Self) -> Self {
        let err = sums(&[emul(self.vb(), o.err), emul(o.vb(), self.err), emul(self.err, o.err)]);
        if self.is_zero() || o.is_zero() { return Self::zero().with_err(err); }
        let mut t = C::w();
        mul_to(self.m.as_ref(), o.m.as_ref(), &mut t.as_mut()[..2 * C::L]);
        let (r, round) = Self::norm(self.neg != o.neg, &t.as_ref()[..2 * C::L], self.e + o.e - 2 * (Self::P - 1));
        r.with_err(sum(err, round))
    }

    /// `self * k` for a small integer (the product rounded once).
    pub fn mul_int(self, k: i64) -> Self { self.mul(Self::from_int(k as i128)) }

    pub fn div(self, o: Self) -> MpfResult<C> {
        if o.is_zero() { return Err(if o.err.is_none() { MpfErr::Domain } else { MpfErr::Undecided }); }
        if let Some(eb) = top(o.err) { if eb > o.e - 1 { return Err(MpfErr::Undecided); } }
        let (r, round) = if self.is_zero() {
            (Self::zero(), None)
        } else {
            // (a << (P + 1)) / b: a quotient of P + 1 or P + 2 bits
            let l = C::L;
            let mut u = C::w();
            shl_to(self.m.as_ref(), Self::P + 1, &mut u.as_mut()[..2 * l + 1]);
            let mut q = C::w();
            let mut un = C::w();
            let mut vn = C::a();
            let rem = divrem(&u.as_ref()[..2 * l + 1], o.m.as_ref(), &mut q.as_mut()[..l + 2], un.as_mut(), vn.as_mut());
            let (r, mut round) = Self::norm(self.neg != o.neg, &q.as_ref()[..l + 2], self.e - o.e - Self::P - 1);
            if rem && round.is_none() { round = p2(r.ulp_exp()); }
            (r, round)
        };
        if r.is_zero() && round.is_none() && self.err.is_none() && o.err.is_none() { return Ok(r); }
        // |a'/b' - a/b| <= (|da| + |a/b| |db|) / |b'|; |b'| >= (t - 1) 2^(e - 31)
        // when |db| is below 2^(e - 32), else >= 2^(e - 1)
        let num = sum(self.err, emul(r.vb(), o.err));
        let tight = match top(o.err) { None => true, Some(t) => t <= o.e - 32 };
        let prop = if tight { ediv(num, o.top32() - 1, o.e - 31) } else { scale(num, -(o.e - 1)) };
        Ok(r.with_err(sum(prop, round)))
    }

    /// Integer square root of a limb slice (Newton from above).
    fn isqrt_limbs(nn: &[u64], out: &mut [u64]) -> bool {
        // out: ceil(len/2) + 1 limbs; returns whether the root is exact
        let bits = bitlen(nn);
        for x in out.iter_mut() { *x = 0; }
        if bits == 0 { return true; }
        // start: (isqrt(top) + 1) 2^(sh/2) >= sqrt(nn), top the leading
        // 126 bits or fewer and sh even
        let mut sh = (bits - 126).max(0);
        if sh % 2 == 1 { sh += 1; }
        let mut tt = [0u64; 2];
        shr_to(nn, sh, &mut tt);
        let top = tt[0] as u128 | ((tt[1] as u128) << 64);
        let r0 = top.isqrt() + 1;
        let r0w = [r0 as u64, (r0 >> 64) as u64];
        shl_to(&r0w, sh / 2, out);
        let len = out.len();
        let mut q = C::w();
        let mut un = C::w();
        let mut vn = C::w();
        let mut next = C::w();
        loop {
            let vl = ((bitlen(out) + 63) / 64) as usize;
            let ul = nn.len();
            if ul < vl { break; }
            divrem(nn, &out[..vl], &mut q.as_mut()[..ul - vl + 1], un.as_mut(), &mut vn.as_mut()[..vl]);
            // next = (out + q) / 2
            let nx = &mut next.as_mut()[..len + 1];
            for x in nx.iter_mut() { *x = 0; }
            nx[..len].copy_from_slice(out);
            add_in(nx, &q.as_ref()[..(ul - vl + 1).min(len + 1)]);
            let mut half = C::w();
            shr_to(nx, 1, &mut half.as_mut()[..len]);
            if cmp_limbs(&half.as_ref()[..len], out) != Ordering::Less { break; }
            out.copy_from_slice(&half.as_ref()[..len]);
        }
        // exact iff out^2 == nn
        let mut sq = C::w();
        let ol = len.min(C::L + 1);
        let sql = (2 * ol).min(2 * C::L + 2);
        mul_to(&out[..ol.min(sql / 2)], &out[..ol.min(sql / 2)], &mut sq.as_mut()[..sql]);
        cmp_limbs(&sq.as_ref()[..sql], nn) == Ordering::Equal
    }

    pub fn sqrt(self) -> MpfResult<C> {
        if self.is_zero() {
            return Ok(match top(self.err) { None => Self::zero(), Some(x) => Self::zero().with_err(p2(x.div_euclid(2) + 1)) });
        }
        if !self.sign_certain() { return Err(MpfErr::Undecided); }
        if self.neg { return Err(MpfErr::Domain); }
        // M 2^s with s = e - P + 1 made even, then N = M' 2^(P + 2): the root has P + 1 bits or more
        let l = C::L;
        let s = self.e - Self::P + 1;
        let (shift, s2) = if s.rem_euclid(2) == 0 { (Self::P + 2, s) } else { (Self::P + 3, s - 1) };
        let mut nn = C::w();
        shl_to(self.m.as_ref(), shift, &mut nn.as_mut()[..2 * l + 1]);
        let mut root = C::w();
        let exact = Self::isqrt_limbs(&nn.as_ref()[..2 * l + 1], &mut root.as_mut()[..l + 2]);
        // value = root * 2^((s2 - (P + 2)) / 2) (s2 - P - 2 even)
        let (r, mut round) = Self::norm(false, &root.as_ref()[..l + 2], (s2 - (Self::P + 2)) / 2);
        if !exact && round.is_none() { round = p2(r.ulp_exp()); }
        // |sqrt(x + d) - sqrt(x)| <= |d| / sqrt(x), sqrt(x) >= 2^floor(e/2)
        let prop = scale(self.err, -self.e.div_euclid(2));
        Ok(r.with_err(sum(prop, round)))
    }

    /// The integer part (toward zero) and whether a fraction was dropped.
    fn trunc(self) -> (Self, bool) {
        if self.is_zero() || self.e < 0 { return (Self::zero(), !self.is_zero()); }
        let drop = Self::P - 1 - self.e;
        if drop <= 0 { return (self.with_err(None), false); }
        let mut m = self.m;
        let had = low_any(m.as_ref(), drop);
        let mut t = C::a();
        shr_to(m.as_ref(), drop, t.as_mut());
        shl_to(t.as_ref(), drop, m.as_mut());
        (Mpf { neg: self.neg, m, e: self.e, err: None }, had)
    }

    /// The low 64 bits of the integer part of `|self|` (for quadrants).
    fn int_low_u64(&self) -> u64 {
        if self.is_zero() || self.e < 0 { return 0; }
        let drop = Self::P - 1 - self.e;
        let mut t = [0u64; 2];
        if drop >= 0 { shr_to(self.m.as_ref(), drop, &mut t); } else { shl_to(self.m.as_ref(), -drop, &mut t); }
        t[0]
    }

    /// `self` as an i64 if it is an integer of at most 62 bits.
    fn to_small_int(&self) -> Option<i64> {
        if self.is_zero() { return Some(0); }
        if self.e > 61 { return None; }
        let (t, frac) = self.trunc();
        if frac { return None; }
        let v = t.int_low_u64() as i64;
        Some(if self.neg { -v } else { v })
    }
}

// ============================================================================
// Constants (computed once per thread and capacity, at that capacity)
// ============================================================================

/// A constant: magnitude limbs (the first `len`), top exponent, bound.
#[derive(Clone, Copy)]
struct Konst { m: [u64; 65], len: usize, e: i64, err: E }

impl Konst {
    fn of<C: Cap>(v: Mpf<C>) -> Konst {
        let mut m = [0u64; 65];
        m[..C::L].copy_from_slice(v.m.as_ref());
        Konst { m, len: C::L, e: v.e, err: v.err }
    }
    fn at<C: Cap>(&self) -> Mpf<C> { Mpf::<C>::from_limbs(false, &self.m[..self.len], self.e, self.err) }
}

/// Per capacity (C2, C3, C5, C9, C17, C33, C65): ln 2, pi/2, gamma.
#[derive(Clone, Copy, Default)]
struct Consts { ln2: Option<Konst>, half_pi: Option<Konst>, gamma: Option<Konst> }

thread_local! {
    static CONSTS: core::cell::RefCell<[Consts; 7]> = core::cell::RefCell::new([Consts::default(); 7]);
}

fn slot(l: usize) -> usize { match l { 2 => 0, 3 => 1, 5 => 2, 9 => 3, 17 => 4, 33 => 5, _ => 6 } }

/// The cached constant of `C`, computed by `make` on first use.
fn cached<C: Cap>(pick: fn(&mut Consts) -> &mut Option<Konst>, make: fn() -> MpfResult<C>) -> MpfResult<C> {
    let have = CONSTS.with(|c| *pick(&mut c.borrow_mut()[slot(C::L)]));
    if let Some(k) = have { return Ok(k.at::<C>()); }
    let v = make()?;
    CONSTS.with(|c| *pick(&mut c.borrow_mut()[slot(C::L)]) = Some(Konst::of(v)));
    Ok(v)
}

/// `atan(1/n)` by its series, with the truncation bound.
fn atan_inv<C: Cap>(n: u64) -> Mpf<C> {
    let mut t = Mpf::<C>::from_int(1).div_u64(n);
    let n2 = n * n;
    let mut s = Mpf::<C>::zero();
    let mut k: u64 = 0;
    loop {
        let term = t.div_u64(2 * k + 1);
        if term.is_zero() || term.e < -Mpf::<C>::P - 8 {
            // alternating, decreasing: the rest is below this term
            return s.with_err(sum(s.err, term.vb()));
        }
        s = if k % 2 == 0 { s.add(term) } else { s.sub(term) };
        t = t.div_u64(n2);
        k += 1;
    }
}

/// ln 2 = 2 sum (1/3)^(2k+1) / (2k+1); the rest below 9/8 of the last term.
fn make_ln2<C: Cap>() -> MpfResult<C> {
    let mut t = Mpf::<C>::from_int(1).div_u64(3);
    let mut s = Mpf::<C>::zero();
    let mut k: u64 = 0;
    loop {
        let term = t.div_u64(2 * k + 1);
        if term.is_zero() || term.e < -Mpf::<C>::P - 8 {
            s = s.with_err(sum(s.err, scale(term.vb(), 1)));
            break;
        }
        s = s.add(term);
        t = t.div_u64(9);
        k += 1;
    }
    Ok(s.shl(1))
}

/// pi/2 = 8 atan(1/5) - 2 atan(1/239).
fn make_half_pi<C: Cap>() -> MpfResult<C> { Ok(atan_inv::<C>(5).shl(3).sub(atan_inv::<C>(239).shl(1))) }

/// Euler's gamma by Brent and McMillan: `gamma = A/B - ln n`,
/// `B = sum (n^k/k!)^2`, `A = sum (n^k/k!)^2 H_k`, method error below
/// `pi e^-4n`, below 2^-(P + 12) for n = (P + 12)/5.
fn make_gamma<C: Cap>() -> MpfResult<C> {
    let n = (Mpf::<C>::P as u64 + 12) / 5 + 1;
    let (mut b, mut h) = (Mpf::<C>::from_int(1), Mpf::<C>::zero());
    let (mut a_sum, mut b_sum) = (Mpf::<C>::zero(), Mpf::<C>::from_int(1));
    let mut k: u64 = 1;
    loop {
        b = b.mul_u64(n * n).div_u64(k * k);
        h = h.add(Mpf::<C>::from_int(1).div_u64(k));
        let a = b.mul(h);
        a_sum = a_sum.add(a);
        b_sum = b_sum.add(b);
        if k > 2 * n && a.e < a_sum.e - Mpf::<C>::P - 8 {
            // past k = 2n each term is below a quarter of the one before
            // (H grows by less than a third): the tails are below twice the last terms
            a_sum = a_sum.with_err(sum(a_sum.err, p2(a.e + 2)));
            b_sum = b_sum.with_err(sum(b_sum.err, p2(b.e + 2)));
            break;
        }
        k += 1;
    }
    let g = a_sum.div(b_sum)?.sub(Mpf::<C>::from_int(n as i128).ln()?);
    Ok(g.with_err(sum(g.err, p2(-(Mpf::<C>::P + 12)))))
}

// ============================================================================
// Functions
// ============================================================================

impl<C: Cap> Mpf<C> {
    /// `self / d` for a small positive divisor (rounded once).
    pub fn div_u64(self, d: u64) -> Self {
        let err = ediv(self.err, d, 0);
        if self.is_zero() { return self.with_err(err); }
        let mut t = C::w();
        t.as_mut()[1..C::L + 1].copy_from_slice(self.m.as_ref());
        let rem = div_small_in(&mut t.as_mut()[..C::L + 1], d);
        let (r, mut round) = Self::norm(self.neg, &t.as_ref()[..C::L + 1], self.e - Self::P + 1 - 64);
        if rem != 0 && round.is_none() { round = p2(r.ulp_exp()); }
        r.with_err(sum(err, round))
    }

    /// `self * k` for a small positive factor (rounded once).
    pub fn mul_u64(self, k: u64) -> Self {
        let err = emul(self.err, Some(bnorm(k as u128, 0)));
        if self.is_zero() { return self.with_err(err); }
        let mut t = C::w();
        t.as_mut()[..C::L].copy_from_slice(self.m.as_ref());
        let c = mul_small_in(&mut t.as_mut()[..C::L], k);
        t.as_mut()[C::L] = c;
        let (r, round) = Self::norm(self.neg, &t.as_ref()[..C::L + 1], self.e - Self::P + 1);
        r.with_err(sum(err, round))
    }

    pub fn ln2() -> Self { cached::<C>(|c| &mut c.ln2, make_ln2::<C>).expect("ln 2") }

    pub fn half_pi() -> Self { cached::<C>(|c| &mut c.half_pi, make_half_pi::<C>).expect("pi") }

    pub fn pi() -> Self { Self::half_pi().shl(1) }

    pub fn euler_gamma() -> MpfResult<C> { cached::<C>(|c| &mut c.gamma, make_gamma::<C>) }

    /// Halvings before a series at this precision (about sqrt P: the series
    /// then needs about sqrt P terms).
    fn halvings() -> i64 { ((Self::P as u64).isqrt() as i64).max(4) }

    pub fn exp(self) -> MpfResult<C> {
        if self.is_zero() && self.err.is_none() { return Ok(Self::from_int(1)); }
        if let Some(t) = top(self.err) { if t > -1 { return Err(MpfErr::Undecided); } }
        if !self.is_zero() && self.e >= 31 {
            if self.neg { return Ok(Self::zero().with_err(p2(-(1i64 << 31)))); }
            return Err(MpfErr::Overflow);
        }
        let x = self.with_err(None);
        let ln2 = Self::ln2();
        // k = floor(x / ln 2), then r = x - k ln 2 in [0, ln 2)
        let q = x.div(ln2.with_err(None))?;
        let (qt, frac) = q.trunc();
        let mut k = qt.to_small_int().ok_or(MpfErr::Overflow)?;
        if q.is_negative() && frac { k -= 1; }
        let mut r = x.sub(ln2.mul_int(k));
        let mut guard = 0;
        while r.is_negative() && guard < 4 { k -= 1; r = r.add(ln2); guard += 1; }
        while !r.sub(ln2).is_negative() && guard < 8 { k += 1; r = r.sub(ln2); guard += 1; }
        if r.is_negative() { r = Self::zero().with_err(sum(r.err, r.vb())); }
        // exp(r) = exp(r / 2^K)^(2^K), the series of the small argument
        let kk = Self::halvings();
        let s = r.shl(-kk);
        let mut acc = Self::from_int(1);
        let mut term = Self::from_int(1);
        let mut j: u64 = 1;
        loop {
            term = term.mul(s).div_u64(j);
            if term.is_zero() || term.e < -Self::P - 4 {
                // |s| < 1/2: the rest is below the last term
                acc = acc.with_err(sum(acc.err, term.vb()));
                break;
            }
            acc = acc.add(term);
            j += 1;
        }
        for _ in 0..kk { acc = acc.mul(acc); }
        let res = acc.shl(k);
        // |e^(x+d) - e^x| <= 2 e^x |d| for |d| <= 1/2
        let prop = emul(scale(res.vb(), 1), self.err);
        Ok(res.with_err(sum(res.err, prop)))
    }

    pub fn ln(self) -> MpfResult<C> {
        if self.is_zero() { return Err(if self.err.is_none() { MpfErr::Domain } else { MpfErr::Undecided }); }
        if !self.sign_certain() { return Err(MpfErr::Undecided); }
        if self.neg { return Err(MpfErr::Domain); }
        let one = Self::from_int(1);
        let x = self.with_err(None);
        if self.err.is_none() && cmp_limbs(x.m.as_ref(), one.m.as_ref()) == Ordering::Equal && x.e == 0 { return Ok(Self::zero()); }
        if let Some(t) = top(self.err) { if t > self.e - 1 { return Err(MpfErr::Undecided); } }
        // x = m 2^e, m in [1, 2)
        let m = Mpf { e: 0, ..x };
        // a start from the Q64.64 engine on the top bits of m (m 2^64 = M >> (P - 65))
        let m64 = {
            let mut t = [0u64; 2];
            shr_to(m.m.as_ref(), Self::P - 65, &mut t);
            (t[0] as u128 | ((t[1] as u128) << 64)) as i128
        };
        let y0 = crate::fixed_point::domains::binary_fixed::transcendental::ln_tier_n_plus_1::ln_q64_64_native(m64);
        let y = Self::from_raw(y0, 64);
        // ln m = y + ln(m e^-y) exactly; m e^-y - 1 is tiny, so ln1p of it
        // is a short series. The bound never trusts the start.
        let t = m.mul(y.neg().exp()?).sub(one);
        let lnm = y.add(t.ln1p()?);
        let res = lnm.add(Self::ln2().mul_int(x.e));
        // |ln(x + d) - ln x| <= 2|d| / x for |d| <= x/2
        let prop = scale(self.err, 1 - self.e);
        Ok(res.with_err(sum(res.err, prop)))
    }

    /// `(sin x, cos x)`.
    pub fn sin_cos(self) -> Result<(Self, Self), MpfErr> {
        if self.is_zero() && self.err.is_none() { return Ok((Self::zero(), Self::from_int(1))); }
        if let Some(t) = top(self.err) { if t > 0 { return Err(MpfErr::Undecided); } }
        if !self.is_zero() && self.e > Self::P - 64 { return Err(MpfErr::Undecided); }
        let ax = self.abs().with_err(None);
        let hp = Self::half_pi();
        // q = floor(|x| / (pi/2)), r = |x| - q pi/2 in [0, pi/2)
        let (mut q, mut r) = if ax.is_zero() {
            (Self::zero(), Self::zero())
        } else {
            let qq = ax.div(hp.with_err(None))?.trunc().0;
            (qq, ax.sub(hp.mul(qq)))
        };
        let one = Self::from_int(1);
        let mut guard = 0;
        while r.is_negative() && guard < 4 { q = q.sub(one); r = r.add(hp); guard += 1; }
        while !r.sub(hp).is_negative() && guard < 8 { q = q.add(one); r = r.sub(hp); guard += 1; }
        if r.is_negative() { r = Self::zero().with_err(sum(r.err, r.vb())); }
        let quad = (q.int_low_u64() & 3) as u32;
        // (sin a, cos a) of a = r / 2^H by the series, then H double angles
        let hh = Self::halvings();
        let a = r.shl(-hh);
        let a2 = a.mul(a);
        let (mut s, mut c) = (a, one);
        let (mut ts, mut tc) = (a, one);
        let mut k: u64 = 1;
        loop {
            ts = ts.mul(a2).div_u64((2 * k) * (2 * k + 1)).neg();
            tc = tc.mul(a2).div_u64((2 * k - 1) * (2 * k)).neg();
            let small = |t: &Self| t.is_zero() || t.e < -Self::P - 4;
            if small(&ts) && small(&tc) {
                // alternating, decreasing: the rests are below these terms
                s = s.with_err(sum(s.err, ts.vb()));
                c = c.with_err(sum(c.err, tc.vb()));
                break;
            }
            s = s.add(ts);
            c = c.add(tc);
            k += 1;
        }
        for _ in 0..hh {
            let s2 = s.mul(c).shl(1);
            let c2 = c.mul(c).sub(s.mul(s));
            s = s2;
            c = c2;
        }
        let (s, c) = match quad { 0 => (s, c), 1 => (c, s.neg()), 2 => (s.neg(), c.neg()), _ => (c.neg(), s) };
        let s = if self.neg { s.neg() } else { s };
        // both are 1-Lipschitz
        Ok((s.with_err(sum(s.err, self.err)), c.with_err(sum(c.err, self.err))))
    }

    /// `atan t` for |t| <= 1 (inputs exact): half-angle steps, the series,
    /// the factor back.
    fn atan_small(t: Self) -> Self {
        let hh = 4 + (Self::P as u64).isqrt() as i64 / 8;
        let one = Self::from_int(1);
        let mut t = t;
        for _ in 0..hh {
            let d = one.add(one.add(t.mul(t)).sqrt().unwrap_or(one));
            t = t.div(d).unwrap_or(t);
        }
        let t2 = t.mul(t);
        let mut acc = t;
        let mut pw = t;
        let mut k: u64 = 1;
        loop {
            pw = pw.mul(t2).neg();
            let term = pw.div_u64(2 * k + 1);
            if term.is_zero() || term.e < -Self::P - 4 {
                acc = acc.with_err(sum(acc.err, term.vb()));
                break;
            }
            acc = acc.add(term);
            k += 1;
        }
        acc.shl(hh)
    }

    /// `atan2(self, x)`, `self` the ordinate.
    pub fn atan2(self, x: Self) -> MpfResult<C> {
        let y = self;
        if y.is_zero() && x.is_zero() {
            return Err(if y.err.is_none() && x.err.is_none() { MpfErr::Domain } else { MpfErr::Undecided });
        }
        let top_e = if y.is_zero() { x.e } else if x.is_zero() { y.e } else { y.e.max(x.e) };
        // the point must stay away from the origin: r >= 2^top
        let d = sum(y.err, x.err);
        if let Some(dv) = top(d) { if dv > top_e - 2 { return Err(MpfErr::Undecided); } }
        let (ye, xe) = (y.with_err(None), x.with_err(None));
        let (ay, ax) = (ye.abs(), xe.abs());
        let hp = Self::half_pi();
        let a = if cmp_mag(&ay, &ax) != Ordering::Greater {
            let base = Self::atan_small(ye.div(xe)?);
            if !xe.is_negative() { base } else if ye.is_negative() { base.sub(hp.shl(1)) } else { base.add(hp.shl(1)) }
        } else {
            let base = Self::atan_small(xe.div(ye)?);
            if ye.is_negative() { hp.neg().sub(base) } else { hp.sub(base) }
        };
        // |d atan2| <= (|y| dx + |x| dy) / r^2, r^2 >= 2^(2 top) (and the
        // perturbation keeps r above half of it)
        let prop = scale(sum(emul(y.vb(), x.err), emul(x.vb(), y.err)), 2 - 2 * top_e);
        Ok(a.with_err(sum(a.err, prop)))
    }

    pub fn atan(self) -> MpfResult<C> { self.atan2(Self::from_int(1)) }

    pub fn tan(self) -> MpfResult<C> { let (s, c) = self.sin_cos()?; s.div(c) }

    /// `1 - x^2` as `(1 - |x|)(1 + |x|)`.
    fn one_minus_sq(self) -> Self {
        let one = Self::from_int(1);
        let ax = self.abs();
        one.sub(ax).mul(one.add(ax))
    }

    pub fn asin(self) -> MpfResult<C> {
        let g = self.one_minus_sq();
        if !g.sign_certain() { return Err(MpfErr::Undecided); }
        if g.is_negative() { return Err(MpfErr::Domain); }
        self.atan2(g.sqrt()?)
    }

    pub fn acos(self) -> MpfResult<C> {
        let g = self.one_minus_sq();
        if !g.sign_certain() { return Err(MpfErr::Undecided); }
        if g.is_negative() { return Err(MpfErr::Domain); }
        g.sqrt()?.atan2(self)
    }

    /// `sinh x` for |x| < 1/8 by its series.
    fn sinh_series(self) -> Self {
        let x2 = self.mul(self);
        let mut term = self;
        let mut acc = self;
        let mut k: u64 = 1;
        loop {
            term = term.mul(x2).div_u64((2 * k) * (2 * k + 1));
            if term.is_zero() || term.e < acc.e - Self::P - 8 {
                return acc.with_err(sum(acc.err, scale(term.vb(), 1)));
            }
            acc = acc.add(term);
            k += 1;
        }
    }

    fn exp_pair(self) -> Result<(Self, Self), MpfErr> {
        let a = self.abs().exp()?;
        let b = Self::from_int(1).div(a)?;
        Ok((a, b))
    }

    pub fn sinh(self) -> MpfResult<C> {
        if self.is_zero() && self.err.is_none() { return Ok(Self::zero()); }
        if !self.is_zero() && self.e < -3 { return Ok(self.sinh_series()); }
        let (a, b) = self.exp_pair()?;
        let v = a.sub(b).shl(-1);
        Ok(if self.neg { v.neg() } else { v })
    }

    pub fn cosh(self) -> MpfResult<C> {
        if self.is_zero() && self.err.is_none() { return Ok(Self::from_int(1)); }
        let (a, b) = self.exp_pair()?;
        Ok(a.add(b).shl(-1))
    }

    pub fn tanh(self) -> MpfResult<C> {
        if self.is_zero() && self.err.is_none() { return Ok(Self::zero()); }
        if !self.is_zero() && self.e >= 31 {
            // 1 - tanh|x| < 2 e^-2|x|
            let one = Self::from_int(1).with_err(sum(self.err, p2(-(1i64 << 31))));
            return Ok(if self.neg { one.neg() } else { one });
        }
        let v = if !self.is_zero() && self.e < -3 {
            let s = self.abs().sinh_series();
            let c = Self::from_int(1).add(s.mul(s)).sqrt()?;
            s.div(c)?
        } else {
            let t = self.abs().shl(1).neg().exp()?;
            let one = Self::from_int(1);
            one.sub(t).div(one.add(t))?
        };
        Ok(if self.neg { v.neg() } else { v })
    }

    /// `ln(1 + u)`: for |u| < 1/8 by `2 atanh(u / (2 + u))` and its series
    /// (no cancellation), else `ln` of the sum.
    pub fn ln1p(self) -> MpfResult<C> {
        let one = Self::from_int(1);
        if self.is_zero() && self.err.is_none() { return Ok(Self::zero()); }
        if self.is_zero() || self.e >= -3 { return one.add(self).ln(); }
        let u = self.with_err(None);
        let w = u.div(Self::from_int(2).add(u))?;
        let w2 = w.mul(w);
        let mut acc = w;
        let mut pw = w;
        let mut k: u64 = 1;
        loop {
            pw = pw.mul(w2);
            let term = pw.div_u64(2 * k + 1);
            if term.is_zero() || term.e < acc.e - Self::P - 8 {
                // |w| < 1/15: the rest is below a tenth of the last term
                acc = acc.with_err(sum(acc.err, term.vb()));
                break;
            }
            acc = acc.add(term);
            k += 1;
        }
        let v = acc.shl(1);
        // |d ln(1 + u)| <= |du| / (1 + u) <= 2 |du| for u > -1/2
        let prop = scale(self.err, 1);
        Ok(v.with_err(sum(v.err, prop)))
    }

    pub fn asinh(self) -> MpfResult<C> {
        // asinh|x| = ln1p(|x| + x^2 / (1 + sqrt(1 + x^2)))
        let ax = self.abs();
        let one = Self::from_int(1);
        let x2 = ax.mul(ax);
        let u = ax.add(x2.div(one.add(one.add(x2).sqrt()?))?);
        let v = u.ln1p()?;
        Ok(if self.neg { v.neg() } else { v })
    }

    pub fn acosh(self) -> MpfResult<C> {
        let one = Self::from_int(1);
        let g = self.sub(one);
        if !g.sign_certain() { return Err(MpfErr::Undecided); }
        if g.is_negative() { return Err(MpfErr::Domain); }
        let r = g.mul(self.add(one)).sqrt()?;
        self.add(r).ln()
    }

    pub fn atanh(self) -> MpfResult<C> {
        let one = Self::from_int(1);
        let g = one.sub(self.abs());
        if g.is_zero() && g.err.is_none() { return Err(MpfErr::Domain); }
        if !g.sign_certain() { return Err(MpfErr::Undecided); }
        if g.is_negative() { return Err(MpfErr::Domain); }
        // atanh x = ln1p(2x / (1 - x)) / 2
        Ok(self.shl(1).div(one.sub(self))?.ln1p()?.shl(-1))
    }

    /// `self^y`: `exp(y ln x)` for x > 0; 0 for an exact zero base and y > 0.
    /// `self^y = exp(y ln x)` for x > 0; a zero or negative base is left to
    /// the caller's contract (`Unsupported`).
    pub fn pow(self, y: Self) -> MpfResult<C> {
        if !self.sign_certain() { return Err(MpfErr::Undecided); }
        if self.is_negative() || self.is_zero() { return Err(MpfErr::Unsupported); }
        if y.is_zero() && y.err.is_none() { return Ok(Self::from_int(1)); }
        y.mul(self.ln()?).exp()
    }

    /// The nearest raw at `f` fraction bits, ties toward +infinity, when
    /// the bound decides it. `tie` rounds a value within a negligible bound
    /// of a tie as the tie.
    pub fn round_to_raw<S: WideInt>(&self, f: u32, tie: bool) -> Result<S, MpfErr> { self.round_to_raw_mode(f, tie, false) }

    /// As [`Mpf::round_to_raw`], with ties to the even raw when `even`
    /// (the decimal domain's rule) instead of toward +infinity.
    pub fn round_to_raw_mode<S: WideInt>(&self, f: u32, tie: bool, even: bool) -> Result<S, MpfErr> {
        let fi = f as i64;
        let errt = top(self.err);
        if self.is_zero() {
            return match errt { None => Ok(S::ZERO), Some(x) if x + fi <= -2 => Ok(S::ZERO), _ => Err(MpfErr::Undecided) };
        }
        // value 2^f = M 2^(e - P + 1 + f): drop s = P - 1 - f - e bits
        let s = Self::P - 1 - fi - self.e;
        if s > Self::P + 2 {
            // |v 2^f| < 2^(e + 1 + f) < 1/4
            return match errt { Some(x) if x + fi > -2 => Err(MpfErr::Undecided), _ => Ok(S::ZERO) };
        }
        let mut mag = [0u64; 20];
        if s <= 0 {
            if let Some(x) = errt { if x + fi > -1 { return Err(MpfErr::Undecided); } }
            if bitlen(self.m.as_ref()) - s > S::BITS as i64 { return Err(MpfErr::Overflow); }
            shl_to(self.m.as_ref(), -s, &mut mag);
        } else {
            // in 2L + 2 limbs: s reaches P + 2, past the mantissa
            let l = C::L + 1;
            let mut m = C::w();
            m.as_mut()[..C::L].copy_from_slice(self.m.as_ref());
            let mut q = C::w();
            shr_to(&m.as_ref()[..l], s, &mut q.as_mut()[..l]);
            // rem - half, as a sign and a magnitude
            let mut rem = m;
            let mut hi = C::w();
            shl_to(&q.as_ref()[..l], s, &mut hi.as_mut()[..l]);
            sub_in(&mut rem.as_mut()[..l], &hi.as_ref()[..l]);
            let mut half = C::w();
            half.as_mut()[((s - 1) / 64) as usize] = 1u64 << ((s - 1) % 64);
            let ord = cmp_limbs(&rem.as_ref()[..l], &half.as_ref()[..l]);
            let diff = match ord {
                Ordering::Less => { let mut d = half; sub_in(&mut d.as_mut()[..l], &rem.as_ref()[..l]); d }
                _ => { let mut d = rem; sub_in(&mut d.as_mut()[..l], &half.as_ref()[..l]); d }
            };
            let q = { let mut a = C::a(); a.as_mut().copy_from_slice(&q.as_ref()[..C::L]); a };
            let odd = q.as_ref()[0] & 1 == 1;
            let tie_up = if even { odd } else { !self.neg };
            let mut up = match ord { Ordering::Greater => true, Ordering::Less => false, Ordering::Equal => tie_up };
            if let Some(x) = errt {
                // decided when |diff| 2^-s exceeds the error 2^f
                let t = x + fi + s;
                // a zero distance (the value on the tie) is never decided by a bound
                let b = bitlen(&diff.as_ref()[..l]);
                if b == 0 || b - 1 < t {
                    // within the bound of the tie: at the widest precision, with
                    // the bound below 2^-256 units, the value is taken as the tie
                    if tie && x + fi < -256 { up = tie_up; } else { return Err(MpfErr::NearTie); }
                }
            }
            // a quotient beyond 19 limbs fits no word
            if bitlen(q.as_ref()) > 64 * 19 { return Err(MpfErr::Overflow); }
            let n = C::L.min(20);
            mag[..n].copy_from_slice(&q.as_ref()[..n]);
            if up { add_in(&mut mag, &[1]); }
        }
        let n = bitlen(&mag);
        let k = S::BITS as i64 - 1;
        let at_min = self.neg && n == k + 1 && {
            let mut z = mag;
            z[(k / 64) as usize] &= !(1u64 << (k % 64));
            all_zero(&z)
        };
        if at_min { return Ok(S::from_i128(-1).shl_bits(S::BITS - 1)); }
        if n > k { return Err(MpfErr::Overflow); }
        let words = ((n + 63) / 64).max(1) as usize;
        let mut out = S::from_i128(mag[words - 1] as i128);
        for i in (0..words - 1).rev() {
            out = out.shl_bits(64).checked_add_wide(S::from_i128(mag[i] as i128)).ok_or(MpfErr::Overflow)?;
        }
        if self.neg { out = out.checked_neg_wide().ok_or(MpfErr::Overflow)?; }
        Ok(out)
    }
}

/// `|a|` against `|b|`.
fn cmp_mag<C: Cap>(a: &Mpf<C>, b: &Mpf<C>) -> Ordering {
    match (a.is_zero(), b.is_zero()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Less,
        (false, true) => Ordering::Greater,
        _ => a.e.cmp(&b.e).then_with(|| cmp_limbs(a.m.as_ref(), b.m.as_ref())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixed_point::domains::binary_fixed::transcendental::big::{Big, BigErr};

    /// A `Big` at C65 (exact: its mantissa has 1001 bits).
    fn from_big(b: &Big) -> Mpf<C65> {
        // Big: m 2^(e - 1000), m with its top bit at 1000
        let (neg, m, e, err) = b.parts();
        let mut limbs = [0u64; 16];
        for i in 0..16 { limbs[i] = m.words[i]; }
        let (r, round) = Mpf::<C65>::norm(neg, &limbs, e - 1000);
        r.with_err(sum(sum(err.and_then(p2), round), None))
    }

    fn widen<C: Cap>(a: &Mpf<C>) -> Mpf<C65> { Mpf::<C65>::from_limbs(a.neg, a.m.as_ref(), a.e, a.err) }

    /// |a - b| within the two bounds; returns the slack of a's bound (bits
    /// above its precision).
    fn agree<C: Cap>(what: &str, a: Result<Mpf<C>, MpfErr>, b: Result<Big, BigErr>) -> i64 {
        let (a, b) = match (a, b) {
            (Ok(a), Ok(b)) => (a, b),
            (Err(_), Err(_)) => return 0,
            (Err(MpfErr::Undecided), Ok(_)) | (Ok(_), Err(BigErr::Undecided)) => return 0,
            (x, y) => panic!("{what}: {:?} vs {:?}", x.map(|_| ()), y.map(|_| ())),
        };
        let (wa, wb) = (widen(&a), from_big(&b));
        let d = wa.sub(wb);
        let bound = sum(a.err, wb.err);
        if !d.is_zero() {
            let t = top(sum(bound, d.err)).expect("an inexact pair");
            assert!(d.e < t, "{what} at {} limbs: |a - b| = 2^{} beyond the bounds 2^{}", C::L, d.e, t);
        }
        let slack = match (top(a.err), a.is_zero()) { (Some(t), false) => t - (a.e - Mpf::<C>::P), _ => 0 };
        if slack > 40 { eprintln!("slack {slack:3} at {} limbs: {what}", C::L); }
        slack
    }

    fn cases() -> Vec<(i128, u32)> {
        // raws at 64 fraction bits, a spread of magnitudes and signs
        let mut v = Vec::new();
        for &(n, f) in &[(1i128, 0u32), (3, 1), (-5, 2), (7, 3), (123456789, 20), (-987654321, 30), (1, 40), (-3, 50),
                         (5, 0), (-17, 0), (40, 0), (1000, 3), (2_000_000, 0), (-2_000_000, 3), (99, 7), (1, 1), (-1, 1), (31, 5)] {
            v.push((n, f));
        }
        v
    }

    macro_rules! check_cap {
        ($C:ty) => {{
            let mut worst = i64::MIN;
            for (n, f) in cases() {
                let a = Mpf::<$C>::from_raw(n, f);
                let b = Big::from_raw(n, f);
                let w = format!("x = {n}/2^{f}");
                let mut s = |x: i64| { if x > worst { worst = x; } };
                s(agree(&format!("exp {w}"), a.exp(), b.exp()));
                s(agree(&format!("ln {w}"), a.ln(), b.ln()));
                s(agree(&format!("sqrt {w}"), a.sqrt(), b.sqrt()));
                s(agree(&format!("sin {w}"), a.sin_cos().map(|p| p.0), b.sin_cos().map(|p| p.0)));
                s(agree(&format!("cos {w}"), a.sin_cos().map(|p| p.1), b.sin_cos().map(|p| p.1)));
                s(agree(&format!("tan {w}"), a.tan(), b.tan()));
                s(agree(&format!("atan {w}"), a.atan(), b.atan()));
                s(agree(&format!("asin {w}"), a.asin(), b.asin()));
                s(agree(&format!("acos {w}"), a.acos(), b.acos()));
                s(agree(&format!("sinh {w}"), a.sinh(), b.sinh()));
                s(agree(&format!("cosh {w}"), a.cosh(), b.cosh()));
                s(agree(&format!("tanh {w}"), a.tanh(), b.tanh()));
                s(agree(&format!("asinh {w}"), a.asinh(), b.asinh()));
                s(agree(&format!("acosh {w}"), a.acosh(), b.acosh()));
                s(agree(&format!("atanh {w}"), a.atanh(), b.atanh()));
                let (y, by) = (Mpf::<$C>::from_raw(3i128, 1), Big::from_raw(3i128, 1));
                s(agree(&format!("pow {w}"), a.pow(y), b.pow(by)));
                s(agree(&format!("atan2 {w}"), a.atan2(y.neg()), b.atan2(by.neg())));
                s(agree(&format!("div {w}"), a.div(y), b.div(by)));
                s(agree(&format!("mul {w}"), Ok(a.mul(y)), Ok(b.mul(by))));
            }
            agree("pi", Ok(Mpf::<$C>::pi()), Ok(Big::pi()));
            agree("gamma", Mpf::<$C>::euler_gamma(), Big::euler_gamma());
            worst
        }};
    }

    #[test]
    fn every_function_agrees_with_the_1000_bit_path_within_its_bounds() {
        let w3 = check_cap!(C3);
        let w9 = check_cap!(C9);
        let w17 = check_cap!(C17);
        // the bounds stay within 64 bits of each precision
        assert!(w3 < 64 && w9 < 64 && w17 < 64, "bound slack {w3} {w9} {w17}");
    }

    #[test]
    fn limb_division_and_roots() {
        // 2^200 / 3, and sqrt of a perfect square at several precisions
        let a = Mpf::<C5>::from_raw(1i128, 0).shl(200);
        let q = a.div(Mpf::<C5>::from_int(3)).unwrap();
        assert!(!q.is_exact());
        let back = q.mul_int(3);
        let d = back.sub(a);
        assert!(d.is_zero() || d.e < a.e - 300);
        let sq = Mpf::<C3>::from_int(12345678987654321i128 * 12345678987654321i128);
        let r = sq.sqrt().unwrap();
        assert!(r.is_exact());
        assert_eq!(r.round_to_raw::<i128>(0, false).unwrap(), 12345678987654321);
        assert_eq!(Mpf::<C33>::from_int(2).sqrt().unwrap().round_to_raw::<i32>(16, false).unwrap(), 92682);
    }

    #[test]
    fn rounding_ties_and_words() {
        assert_eq!(Mpf::<C3>::from_raw(15i32, 3).round_to_raw::<i32>(2, false).unwrap(), 8);
        assert_eq!(Mpf::<C3>::from_raw(-15i32, 3).round_to_raw::<i32>(2, false).unwrap(), -7);
        assert_eq!(Mpf::<C3>::from_raw(i32::MIN, 0).round_to_raw::<i32>(0, false).unwrap(), i32::MIN);
        assert_eq!(Mpf::<C3>::from_raw(i128::MIN, 0).round_to_raw::<i128>(0, false).unwrap(), i128::MIN);
        assert_eq!(Mpf::<C3>::from_int(1).shl(31).round_to_raw::<i32>(0, false).unwrap_err(), MpfErr::Overflow);
        // values whose discarded part reaches past the mantissa (s = P + 1, P + 2):
        // just below half a unit rounds to 0, a bound reaching the half is undecided
        let tiny = Mpf::<C2>::from_int(1).div(Mpf::<C2>::from_int(3)).unwrap().shl(-11);
        assert_eq!(tiny.round_to_raw::<i32>(10, false).unwrap(), 0);
        assert_eq!(tiny.neg().round_to_raw::<i32>(10, false).unwrap(), 0);
        let near_half = Mpf::<C2>::from_int(1).shl(-11).sub(Mpf::<C2>::from_int(1).shl(-120));
        assert_eq!(near_half.round_to_raw::<i32>(10, false).unwrap(), 0);
        // on the tie with a bound: undecided, never rounded as if exact
        let on_tie = Mpf::<C2>::from_int(1).shl(-11).sub(Mpf::<C2>::from_int(1).shl(-200));
        assert_eq!(on_tie.round_to_raw::<i32>(10, false).unwrap_err(), MpfErr::NearTie);
        let third = Mpf::<C3>::from_int(1).div(Mpf::<C3>::from_int(3)).unwrap();
        assert_eq!(third.round_to_raw::<i64>(32, false).unwrap(), 1431655765);
        // exp(1) at Q64.64 from every capacity
        assert_eq!(Mpf::<C3>::from_int(1).exp().unwrap().round_to_raw::<i128>(64, false).unwrap(), 50143449209799256683);
        assert_eq!(Mpf::<C65>::from_int(1).exp().unwrap().round_to_raw::<i128>(64, false).unwrap(), 50143449209799256683);
    }
}

#[cfg(test)]
mod timing {
    use super::*;
    use std::time::Instant;

    fn t<F: FnMut() -> R, R>(name: &str, n: usize, mut f: F) {
        let s = Instant::now();
        for _ in 0..n { std::hint::black_box(f()); }
        eprintln!("{name:24} {:8.0} ns", s.elapsed().as_nanos() as f64 / n as f64);
    }

    macro_rules! cap_times {
        ($C:ty) => {{
            let x = Mpf::<$C>::from_raw(123456789i64, 27);
            let y = Mpf::<$C>::from_raw(-98765i64, 13);
            eprintln!("-- {} limbs", <$C>::L);
            t("add", 20000, || x.add(y));
            t("mul", 20000, || x.mul(y));
            t("div", 20000, || x.div(y));
            t("div_u64", 20000, || x.div_u64(7));
            t("sqrt", 5000, || x.sqrt());
            t("exp", 2000, || x.exp());
            t("ln", 2000, || x.ln());
            t("sin_cos", 2000, || x.sin_cos());
            t("atan", 2000, || x.atan());
        }};
    }

    #[test]
    #[ignore]
    fn primitive_costs() {
        cap_times!(C2);
        cap_times!(C3);
        cap_times!(C5);
        cap_times!(C9);
    }
}
