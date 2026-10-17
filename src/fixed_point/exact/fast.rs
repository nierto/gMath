//! The first stage of the exact tree evaluation: the format's compute tier
//! (2F fraction bits, the engines the imperative methods use) with an error
//! bound carried through every node. Most trees are decided here at about
//! the cost of the compute-tier evaluation; anything undecided, out of the
//! engines' ranges, or touching a domain edge goes to `Mpf`.
//!
//! The engines' bounds are the imperative ladder's margins (`Fixed::settle`):
//! `((|r| + 1) 2^(rel + extra) + 2^2)` compute units, `rel` 1 on the 32-bit
//! word and 6 on the wider ones, `extra` 0 for exp, 2 for ln and sqrt, 4 for
//! atan and atan2, and for sin and cos the argument's integer bits plus 4 on
//! the wider words. They are measured, not proven: the evidence is the
//! calibration test below (compute-precision arguments at every format), the
//! full-range corpora and `tests/chain_correct_rounding_validation.rs`.

use super::Num;
use crate::fixed_point::domains::binary_fixed::transcendental::mpf::{emul, p2, scale, sum, sums, top, MpfErr, E};
use crate::fixed_point::fixed_num::{Compute, FixedNum};
use crate::fixed_point::width::WideInt;

type R<X> = Result<X, MpfErr>;

/// A compute-tier value (`v * 2^-2F`) and a bound on its distance from the
/// exact value it stands for.
pub(crate) struct Fast<T: FixedNum> { v: Compute<T>, err: E }

impl<T: FixedNum> Clone for Fast<T> { fn clone(&self) -> Self { *self } }
impl<T: FixedNum> Copy for Fast<T> {}

impl<T: FixedNum> Fast<T> {
    /// Fraction bits of the compute tier.
    fn cf() -> i64 { 2 * T::FRAC_BITS as i64 }

    fn mag_bits(&self) -> i64 {
        match self.v.checked_neg_wide() {
            Some(n) => (if self.v.is_negative() { n } else { self.v }).bit_length() as i64,
            None => <Compute<T> as WideInt>::BITS as i64,
        }
    }

    /// `|v| < 2^(mag_bits - 2F)`.
    fn vb(&self) -> E { if self.is_zero() { None } else { p2(self.mag_bits() - Self::cf()) } }

    /// `|v| >= 2^low()` (nonzero values).
    fn low(&self) -> i64 { self.mag_bits() - 1 - Self::cf() }

    fn is_zero(&self) -> bool { self.v == <Compute<T> as WideInt>::ZERO }

    fn sign_certain(&self) -> bool {
        if self.is_zero() { return self.err.is_none(); }
        match top(self.err) { None => true, Some(t) => t <= self.low() }
    }

    /// Half a compute unit (a nearest rounding).
    fn half_unit() -> E { p2(-Self::cf() - 1) }

    /// The engine bound for a result `r`: `((|r| + 1) 2^(rel + extra) + 4)` units.
    fn engine(r: &Self, extra: i64) -> E {
        let int_bits = (r.mag_bits() - Self::cf()).max(0) + 1;
        sum(p2(int_bits + Self::rel() + extra - Self::cf()), p2(2 - Self::cf()))
    }

    /// The engines' relative bound, log2 (`Fixed::engine_rel_bits`).
    fn rel() -> i64 { if <T::Storage as WideInt>::BITS == 32 { 1 } else { 6 } }

    fn of(v: Compute<T>, err: E) -> Self { Fast { v, err } }

    fn from_shifted<S: WideInt>(raw: S, f: u32) -> R<Self> {
        // raw * 2^-f at 2F fraction bits, exact
        let cf = Self::cf() as u32;
        if f > cf { return Err(MpfErr::Undecided); }
        let neg = raw.is_negative();
        let mag = if neg { raw.checked_neg_wide().ok_or(MpfErr::Undecided)? } else { raw };
        let bits = mag.bit_length() + (cf - f);
        if bits + 1 >= <Compute<T> as WideInt>::BITS { return Err(MpfErr::Undecided); }
        // the significant words only: a value above 2^64 implies a compute word above 64 bits
        let mut c = <Compute<T> as WideInt>::ZERO;
        let words = ((mag.bit_length() + 63) / 64).max(1);
        for i in (0..words).rev() {
            if i + 1 != words { c = c.shl_bits(64); }
            let w = mag.word(i);
            c = c + <Compute<T> as WideInt>::from_i128((w >> 32) as i128).shl_bits(32) + <Compute<T> as WideInt>::from_i128((w & 0xffff_ffff) as i128);
        }
        let c = c.shl_bits(cf - f);
        Ok(Fast { v: if neg { -c } else { c }, err: None })
    }

    fn int(k: i128) -> R<Self> { Self::from_shifted(k, 0) }

    fn sin_cos(self) -> R<(Self, Self)> {
        if self.is_zero() && self.err.is_none() { return Ok((Fast::of(<Compute<T> as WideInt>::ZERO, None), Self::int(1)?)); }
        if let Some(t) = top(self.err) { if t > 0 { return Err(MpfErr::Undecided); } }
        let arg_int = (self.mag_bits() - Self::cf()).max(0) + 1;
        if arg_int > 25 { return Err(MpfErr::Undecided); }
        let s = Fast::<T>::of(T::direct_sin(self.v), None);
        let c = Fast::<T>::of(T::direct_cos(self.v), None);
        let extra = if <T::Storage as WideInt>::BITS == 32 { 0 } else { arg_int + 4 };
        let es = Self::engine(&s, extra);
        let ec = Self::engine(&c, extra);
        Ok((Fast::of(s.v, sum(es, self.err)), Fast::of(c.v, sum(ec, self.err))))
    }
}

impl<T: FixedNum> Num<T> for Fast<T> {
    fn from_word<W: WideInt>(w: W, f: u32) -> R<Self> { Self::from_shifted(w, f) }

    fn int(k: i128) -> R<Self> { Self::int(k) }

    fn pi() -> R<Self> { Ok(Fast::of(T::compute_pi_half().shl_bits(1), p2(1 - Self::cf()))) }

    fn euler_gamma() -> R<Self> { Err(MpfErr::Undecided) }

    fn neg(self) -> Self { Fast::of(-self.v, self.err) }

    fn half(self) -> Self {
        // exact unless the low bit is set: then half a unit
        let odd = self.v.bit(0);
        Fast::of(self.v.shr_bits(1), sum(scale(self.err, -1), if odd { Self::half_unit() } else { None }))
    }

    fn add(self, o: Self) -> R<Self> {
        let v = self.v.checked_add_wide(o.v).ok_or(MpfErr::Undecided)?;
        Ok(Fast::of(v, sum(self.err, o.err)))
    }

    fn sub(self, o: Self) -> R<Self> {
        let v = self.v.checked_sub_wide(o.v).ok_or(MpfErr::Undecided)?;
        Ok(Fast::of(v, sum(self.err, o.err)))
    }

    fn mul(self, o: Self) -> R<Self> {
        let v = T::compute_checked_multiply(self.v, o.v).map_err(|_| MpfErr::Undecided)?;
        let err = sums(&[emul(self.vb(), o.err), emul(o.vb(), self.err), emul(self.err, o.err), Self::half_unit()]);
        Ok(Fast::of(v, err))
    }

    fn div(self, o: Self) -> R<Self> {
        if o.is_zero() { return Err(MpfErr::Undecided); }
        if let Some(t) = top(o.err) { if t > o.low() - 1 { return Err(MpfErr::Undecided); } }
        let v = T::compute_checked_divide(self.v, o.v).map_err(|_| MpfErr::Undecided)?;
        let q = Fast::<T>::of(v, None);
        let prop = scale(sum(self.err, emul(sum(q.vb(), p2(-Self::cf())), o.err)), -(o.low() - 1));
        Ok(Fast::of(v, sum(prop, Self::half_unit())))
    }

    fn sqrt(self) -> R<Self> {
        if self.is_zero() && self.err.is_none() { return Ok(self); }
        if !self.sign_certain() || self.is_zero() { return Err(MpfErr::Undecided); }
        if self.v.is_negative() { return Err(MpfErr::Undecided); }
        let r = Fast::<T>::of(T::direct_sqrt(self.v), None);
        let prop = scale(self.err, -self.low().div_euclid(2));
        Ok(Fast::of(r.v, sum(Self::engine(&r, 2), prop)))
    }

    fn exp(self) -> R<Self> {
        if let Some(t) = top(self.err) { if t > -1 { return Err(MpfErr::Undecided); } }
        let v = T::direct_exp(self.v);
        if T::exp_sentinel_reached(&v) { return Err(MpfErr::Undecided); }
        let r = Fast::<T>::of(v, None);
        let prop = emul(scale(sum(r.vb(), p2(-Self::cf())), 1), self.err);
        Ok(Fast::of(v, sum(Self::engine(&r, 0), prop)))
    }

    fn ln(self) -> R<Self> {
        if self.is_zero() || !self.sign_certain() || self.v.is_negative() { return Err(MpfErr::Undecided); }
        if let Some(t) = top(self.err) { if t > self.low() - 1 { return Err(MpfErr::Undecided); } }
        let r = Fast::<T>::of(T::direct_ln(self.v), None);
        let prop = scale(self.err, 1 - self.low());
        Ok(Fast::of(r.v, sum(Self::engine(&r, 2), prop)))
    }

    fn sin(self) -> R<Self> { Ok(self.sin_cos()?.0) }

    fn cos(self) -> R<Self> { Ok(self.sin_cos()?.1) }

    fn tan(self) -> R<Self> { let (s, c) = self.sin_cos()?; s.div(c) }

    fn atan2(self, x: Self) -> R<Self> {
        let y = self;
        if y.is_zero() && x.is_zero() { return Err(MpfErr::Undecided); }
        let top_e = if y.is_zero() { x.low() } else if x.is_zero() { y.low() } else { y.low().max(x.low()) };
        let d = sum(y.err, x.err);
        if let Some(t) = top(d) { if t > top_e - 2 { return Err(MpfErr::Undecided); } }
        let r = Fast::<T>::of(T::direct_atan2(y.v, x.v), None);
        let prop = scale(sum(emul(sum(y.vb(), p2(-Self::cf())), x.err), emul(sum(x.vb(), p2(-Self::cf())), y.err)), 2 - 2 * top_e);
        Ok(Fast::of(r.v, sum(Self::engine(&r, 4), prop)))
    }

    fn sign_certain(&self) -> bool { Fast::sign_certain(self) }

    fn is_negative(&self) -> bool { self.v.is_negative() }

    fn is_exact_zero(&self) -> bool { self.is_zero() && self.err.is_none() }

    fn round_int_even(&self, _tie: bool) -> R<T::Storage> {
        // the discarded 2F bits against one half, ties to even only when exact
        let cf = Self::cf() as u32;
        let c = self.v;
        let neg = c.is_negative();
        let mag = if neg { c.checked_neg_wide().ok_or(MpfErr::Undecided)? } else { c };
        let one = <Compute<T> as WideInt>::one();
        let q = mag.shr_bits(cf);
        let below = mag - q.shl_bits(cf);
        let half = one.shl_bits(cf - 1);
        let dist = if below < half { half - below } else { below - half };
        if let Some(t) = top(self.err) {
            let t_units = t + Self::cf();
            if dist == <Compute<T> as WideInt>::ZERO { return Err(MpfErr::Undecided); }
            if t_units >= 0 {
                if t_units as u32 + 2 >= <Compute<T> as WideInt>::BITS { return Err(MpfErr::Undecided); }
                if !(dist > one.shl_bits(t_units as u32)) { return Err(MpfErr::Undecided); }
            }
        }
        let up = below > half || (below == half && q.bit(0));
        let r = if up { q + one } else { q };
        // into the storage word
        let bits = r.bit_length();
        if bits + 1 > <T::Storage as WideInt>::BITS { return Err(MpfErr::Undecided); }
        let mut out = <T::Storage as WideInt>::ZERO;
        let words = ((bits + 63) / 64).max(1);
        for i in (0..words).rev() {
            if i + 1 != words { out = out.shl_bits(64); }
            let w = r.word(i);
            out = out + <T::Storage as WideInt>::from_i128((w >> 32) as i128).shl_bits(32) + <T::Storage as WideInt>::from_i128((w & 0xffff_ffff) as i128);
        }
        Ok(if neg { -out } else { out })
    }

    fn round(&self, _tie: bool) -> R<T::Storage> {
        // the discarded F bits against one half, with the bound in compute units
        let f = T::FRAC_BITS;
        let c = self.v;
        let one = <Compute<T> as WideInt>::one();
        let floor = c.shr_bits(f).shl_bits(f);
        let below = c - floor;
        let half = one.shl_bits(f - 1);
        let dist = if below < half { half - below } else { below - half };
        if let Some(t) = top(self.err) {
            // decided when dist > 2^(t + 2F)
            let t_units = t + Self::cf();
            if t_units >= 0 {
                if t_units as u32 + 2 >= <Compute<T> as WideInt>::BITS { return Err(MpfErr::Undecided); }
                if !(dist > one.shl_bits(t_units as u32)) { return Err(MpfErr::Undecided); }
            } else if dist == <Compute<T> as WideInt>::ZERO {
                return Err(MpfErr::Undecided);
            }
        }
        T::downscale_to_storage(c).map_err(|_| MpfErr::Undecided)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixed_point::domains::binary_fixed::transcendental::mpf::{Mpf, C17};
    use crate::fixed_point::{Fixed, I256, I512};

    /// xorshift64*: a reproducible stream without dependencies.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 { self.0 ^= self.0 >> 12; self.0 ^= self.0 << 25; self.0 ^= self.0 >> 27; self.0.wrapping_mul(0x2545F4914F6CDD1D) }
    }

    /// A compute value with `bits` significant bits below the point shift
    /// (|x| below 2^(bits - 2F)), random sign.
    fn arg<T: FixedNum>(r: &mut Rng, bits: u32) -> Compute<T> {
        let mut c = <Compute<T> as WideInt>::ZERO;
        let words = (bits + 63) / 64;
        for _ in 0..words { c = c.shl_bits(if <Compute<T> as WideInt>::BITS > 64 { 64 } else { 0 }) + <Compute<T> as WideInt>::from_i128((r.next() >> 2) as i128); }
        let keep = bits.min(<Compute<T> as WideInt>::BITS - 2);
        let c = c.shl_bits(<Compute<T> as WideInt>::BITS - 2 - keep.min(<Compute<T> as WideInt>::BITS - 2)).shr_bits(<Compute<T> as WideInt>::BITS - 2 - keep);
        if r.next() & 1 == 1 { -c } else { c }
    }

    /// |fast - exact| within the fast bound, for one function on many
    /// arguments: (checked, decided, worst error in (|r| + 1) compute units, log2).
    fn check<T: FixedNum>(name: &str, n: usize, max_bits: u32, f: impl Fn(Fast<T>) -> R<Fast<T>>, g: impl Fn(Mpf<C17>) -> Result<Mpf<C17>, MpfErr>) -> (usize, usize, i64) {
        let cf = 2 * T::FRAC_BITS;
        let mut r = Rng(0x9E3779B97F4A7C15 ^ (cf as u64) ^ (name.len() as u64) << 40);
        let (mut tried, mut decided, mut worst) = (0, 0, i64::MIN);
        for i in 0..n {
            let bits = 1 + (i as u32 * 7919) % max_bits;
            let x = arg::<T>(&mut r, bits);
            let fx = Fast::<T>::of(x, None);
            let Ok(fr) = f(fx) else { continue };
            let Ok(exact) = g(Mpf::<C17>::from_raw(x, cf)) else { continue };
            tried += 1;
            let got = Mpf::<C17>::from_raw(fr.v, cf);
            let d = got.sub(exact);
            if !d.is_zero() {
                let unit = (fr.mag_bits() - cf as i64).max(0) + 1 - cf as i64;
                worst = worst.max(d.exponent() + 1 - unit);
            }
            if CALIBRATE { if fr.round(false).is_ok() { decided += 1; } continue; }
            let bound = sum(fr.err, exact.err());
            if !d.is_zero() {
                let t = top(sum(bound, d.err())).expect("inexact");
                assert!(d.exponent() < t, "{name} Q.{}: x raw {:?}: |fast - exact| = 2^{} beyond the bound 2^{}", T::FRAC_BITS, x, d.exponent(), t);
            }
            if fr.round(false).is_ok() { decided += 1; }
        }
        (tried, decided, worst)
    }

    /// Report the worst errors instead of asserting the bounds.
    const CALIBRATE: bool = false;

    fn all<T: FixedNum>() {
        let cf = 2 * T::FRAC_BITS;
        let ib = <Compute<T> as WideInt>::BITS - cf - 2;
        let n = 3000;
        let rep = |name: &str, (t, d, w): (usize, usize, i64)| eprintln!("Q.{:<3} {name:6} checked {t:5} decided {d:5} worst 2^{w}", T::FRAC_BITS);
        rep("exp", check::<T>("exp", n, cf + 6.min(ib), |x| x.exp(), |x| x.exp()));
        rep("ln", check::<T>("ln", n, cf + ib, |x| x.abs().ln(), |x| x.abs().ln()));
        rep("sqrt", check::<T>("sqrt", n, cf + ib, |x| x.abs().sqrt(), |x| x.abs().sqrt()));
        rep("sin", check::<T>("sin", n, cf + 20.min(ib), |x| x.sin(), |x| Ok(x.sin_cos()?.0)));
        rep("cos", check::<T>("cos", n, cf + 20.min(ib), |x| x.cos(), |x| Ok(x.sin_cos()?.1)));
        rep("atan", check::<T>("atan", n, cf + ib, |x| x.atan(), |x| x.atan()));
        rep("tan", check::<T>("tan", n, cf + 4, |x| x.tan(), |x| x.tan()));
        rep("tanh", check::<T>("tanh", n, cf + 5, |x| x.tanh(), |x| x.tanh()));
        rep("asinh", check::<T>("asinh", n, cf + ib / 2, |x| x.asinh(), |x| x.asinh()));
        rep("asin", check::<T>("asin", n, cf, |x| x.asin(), |x| x.asin()));
        rep("atan2", check::<T>("atan2", n, cf + ib / 2, |x| x.atan2(Fast::<T>::of(T::compute_pi_half(), None)), |x| x.atan2(Mpf::<C17>::from_raw(T::compute_pi_half(), cf))));
    }

    #[test]
    #[ignore]
    fn realtime_splits_calibration() {
        all::<Fixed<i32, 6>>();
        all::<Fixed<i32, 8>>();
        all::<Fixed<i32, 12>>();
        all::<Fixed<i32, 20>>();
        all::<Fixed<i32, 24>>();
        all::<Fixed<i32, 28>>();
    }

    #[test]
    fn engine_bounds_hold_on_compute_tier_arguments() {
        all::<Fixed<i32, 10>>();
        all::<Fixed<i32, 16>>();
        all::<Fixed<i64, 32>>();
        all::<Fixed<i128, 64>>();
        all::<Fixed<I256, 128>>();
        all::<Fixed<I512, 256>>();
    }
}
