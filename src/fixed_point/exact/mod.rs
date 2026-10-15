//! Correct rounding by two stages (`docs/design/CORRECT_ROUNDING.md`): a
//! formula is written once over [`Num`] and evaluated on the format's compute
//! tier with tracked bounds ([`fast::Fast`]); when a result's rounding is not
//! decided there, it is evaluated again on `Mpf` from just above the format's
//! width up to 4160 bits, where a value within a negligible bound of a tie
//! is taken as the tie.

pub(crate) mod fast;

use crate::fixed_point::domains::binary_fixed::transcendental::mpf::{Cap, Mpf, MpfErr, C17, C2, C3, C33, C5, C65, C9};
use crate::fixed_point::fixed_num::FixedNum;
use crate::fixed_point::width::WideInt;


type R<X> = Result<X, MpfErr>;

/// The arithmetic the tree walker needs: the compute tier with a bound
/// (`fast.rs`) and `Mpf` at a capacity. The composed functions default to
/// formulas on the primitives; `Mpf` overrides them with its
/// cancellation-free forms. A backend that cannot decide something answers
/// `Undecided`; only `Mpf` reports `Domain`.
pub(crate) trait Num<T: FixedNum>: Copy + Sized {
    /// The exact value `w * 2^-f` of a word (an approximation with its
    /// bound where the backend cannot hold it, or `Undecided`).
    fn from_word<W: WideInt>(w: W, f: u32) -> R<Self>;
    fn int(k: i128) -> R<Self> { Self::from_word(k, 0) }
    fn pi() -> R<Self>;
    fn euler_gamma() -> R<Self>;
    fn neg(self) -> Self;
    fn half(self) -> Self;
    fn add(self, o: Self) -> R<Self>;
    fn sub(self, o: Self) -> R<Self>;
    fn mul(self, o: Self) -> R<Self>;
    fn div(self, o: Self) -> R<Self>;
    fn sqrt(self) -> R<Self>;
    fn exp(self) -> R<Self>;
    fn ln(self) -> R<Self>;
    fn sin(self) -> R<Self>;
    fn cos(self) -> R<Self>;
    fn tan(self) -> R<Self>;
    fn atan2(self, x: Self) -> R<Self>;
    fn sign_certain(&self) -> bool;
    fn is_negative(&self) -> bool;
    fn is_exact_zero(&self) -> bool;
    fn round(&self, tie: bool) -> R<T::Storage>;
    /// The nearest integer, ties to even, as a storage word (the decimal
    /// domain's rounding of a value already scaled by 10^dp).
    fn round_int_even(&self, tie: bool) -> R<T::Storage>;

    fn abs(self) -> Self { if self.is_negative() { self.neg() } else { self } }
    fn atan(self) -> R<Self> { self.atan2(Self::int(1)?) }
    /// `(1 - |x|)(1 + |x|)`, which must be certainly non-negative.
    fn one_minus_sq(self) -> R<Self> {
        let one = Self::int(1)?;
        let ax = self.abs();
        let g = one.sub(ax)?.mul(one.add(ax)?)?;
        if !g.sign_certain() || g.is_negative() { return Err(MpfErr::Undecided); }
        Ok(g)
    }
    fn asin(self) -> R<Self> { let g = self.one_minus_sq()?; self.atan2(g.sqrt()?) }
    fn acos(self) -> R<Self> { let g = self.one_minus_sq()?; g.sqrt()?.atan2(self) }
    fn sinh(self) -> R<Self> {
        let a = self.abs().exp()?;
        let v = a.sub(Self::int(1)?.div(a)?)?.half();
        Ok(if self.is_negative() { v.neg() } else { v })
    }
    fn cosh(self) -> R<Self> { let a = self.abs().exp()?; Ok(a.add(Self::int(1)?.div(a)?)?.half()) }
    fn tanh(self) -> R<Self> {
        let ax = self.abs();
        let t = ax.add(ax)?.neg().exp()?;
        let one = Self::int(1)?;
        let v = one.sub(t)?.div(one.add(t)?)?;
        Ok(if self.is_negative() { v.neg() } else { v })
    }
    fn asinh(self) -> R<Self> {
        let ax = self.abs();
        let v = ax.add(ax.mul(ax)?.add(Self::int(1)?)?.sqrt()?)?.ln()?;
        Ok(if self.is_negative() { v.neg() } else { v })
    }
    fn acosh(self) -> R<Self> {
        let one = Self::int(1)?;
        let g = self.sub(one)?;
        if !g.sign_certain() || g.is_negative() { return Err(MpfErr::Undecided); }
        self.add(g.mul(self.add(one)?)?.sqrt()?)?.ln()
    }
    fn atanh(self) -> R<Self> {
        let one = Self::int(1)?;
        let g = one.sub(self.abs())?;
        if !g.sign_certain() || g.is_negative() || g.is_exact_zero() { return Err(MpfErr::Undecided); }
        Ok(one.add(self)?.ln()?.sub(one.sub(self)?.ln()?)?.half())
    }
    /// `x^y = exp(y ln x)` for x > 0; a base that may be zero or negative is
    /// left to the caller's contract (`Unsupported`).
    fn pow(self, y: Self) -> R<Self> {
        if !self.sign_certain() { return Err(MpfErr::Undecided); }
        if self.is_negative() || self.is_exact_zero() { return Err(MpfErr::Unsupported); }
        if y.is_exact_zero() { return Self::int(1); }
        y.mul(self.ln()?)?.exp()
    }
}

impl<T: FixedNum, C: Cap> Num<T> for Mpf<C> {
    fn from_word<W: WideInt>(w: W, f: u32) -> R<Self> { Ok(Mpf::<C>::from_raw(w, f)) }
    fn int(k: i128) -> R<Self> { Ok(Mpf::<C>::from_int(k)) }
    fn pi() -> R<Self> { Ok(Mpf::<C>::pi()) }
    fn euler_gamma() -> R<Self> { Mpf::<C>::euler_gamma() }
    fn neg(self) -> Self { Mpf::neg(self) }
    fn half(self) -> Self { self.shl(-1) }
    fn add(self, o: Self) -> R<Self> { Ok(Mpf::add(self, o)) }
    fn sub(self, o: Self) -> R<Self> { Ok(Mpf::sub(self, o)) }
    fn mul(self, o: Self) -> R<Self> { Ok(Mpf::mul(self, o)) }
    fn div(self, o: Self) -> R<Self> { Mpf::div(self, o) }
    fn sqrt(self) -> R<Self> { Mpf::sqrt(self) }
    fn exp(self) -> R<Self> { Mpf::exp(self) }
    fn ln(self) -> R<Self> { Mpf::ln(self) }
    fn sin(self) -> R<Self> { Ok(self.sin_cos()?.0) }
    fn cos(self) -> R<Self> { Ok(self.sin_cos()?.1) }
    fn tan(self) -> R<Self> { Mpf::tan(self) }
    fn atan2(self, x: Self) -> R<Self> { Mpf::atan2(self, x) }
    fn sign_certain(&self) -> bool { Mpf::sign_certain(self) }
    fn is_negative(&self) -> bool { Mpf::is_negative(self) }
    fn is_exact_zero(&self) -> bool { self.is_zero() && self.is_exact() }
    fn round(&self, tie: bool) -> R<T::Storage> { self.round_to_raw::<T::Storage>(T::FRAC_BITS, tie) }
    fn round_int_even(&self, tie: bool) -> R<T::Storage> { self.round_to_raw_mode::<T::Storage>(0, tie, true) }
    fn atan(self) -> R<Self> { Mpf::atan(self) }
    fn asin(self) -> R<Self> { Mpf::asin(self) }
    fn acos(self) -> R<Self> { Mpf::acos(self) }
    fn sinh(self) -> R<Self> { Mpf::sinh(self) }
    fn cosh(self) -> R<Self> { Mpf::cosh(self) }
    fn tanh(self) -> R<Self> { Mpf::tanh(self) }
    fn asinh(self) -> R<Self> { Mpf::asinh(self) }
    fn acosh(self) -> R<Self> { Mpf::acosh(self) }
    fn atanh(self) -> R<Self> { Mpf::atanh(self) }
    fn pow(self, y: Self) -> R<Self> { Mpf::pow(self, y) }
}

/// A formula with one or more outputs, written once for every backend.
pub(crate) trait Formula<T: FixedNum> {
    fn eval<B: Num<T>>(&self) -> R<Vec<B>>;
}

/// Every output of `g` rounded once to storage: the compute tier first, then
/// `Mpf` at growing capacities while any output is undecided. `Domain` and
/// `Overflow` come from `Mpf` only.
pub(crate) fn settle<T: FixedNum, G: Formula<T>>(g: &G) -> R<Vec<T::Storage>> {
    fn all<T: FixedNum, B: Num<T>>(v: R<Vec<B>>, tie: bool) -> R<Vec<T::Storage>> {
        v?.iter().map(|x| x.round(tie)).collect()
    }
    if let Ok(r) = all::<T, fast::Fast<T>>(g.eval::<fast::Fast<T>>(), false) { return Ok(r); }
    let bits = <T::Storage as WideInt>::BITS;
    macro_rules! attempt {
        ($C:ty, $tie:expr) => {
            match all::<T, Mpf<$C>>(g.eval::<Mpf<$C>>(), $tie) {
                Ok(r) => return Ok(r),
                Err(MpfErr::Undecided) | Err(MpfErr::NearTie) => {}
                Err(x) => return Err(x),
            }
        };
    }
    if bits <= 64 { attempt!(C2, false); }
    if bits <= 128 { attempt!(C3, false); }
    if bits <= 256 { attempt!(C5, false); }
    attempt!(C9, false);
    attempt!(C17, false);
    attempt!(C33, false);
    attempt!(C65, true);
    Err(MpfErr::Undecided)
}
