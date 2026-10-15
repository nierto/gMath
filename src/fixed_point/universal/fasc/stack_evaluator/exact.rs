//! The exact evaluation of a canonical tree: every leaf at its exact value,
//! every node on a backend of `crate::fixed_point::exact` (the compute tier
//! with tracked bounds first, then `Mpf` at growing precision), and one
//! rounding to storage when the bound decides it
//! (`docs/design/CORRECT_ROUNDING.md`).

use super::{FascFormat, LazyExpr, StackEvaluator, StackValue};
use crate::fixed_point::domains::binary_fixed::transcendental::mpf::{Mpf, MpfErr, C17, C2, C3, C33, C5, C65, C9};
use crate::fixed_point::domains::symbolic::rational::rational_number::{RationalNumber, RationalParts};
use crate::fixed_point::exact::{fast::Fast, Num};
use crate::fixed_point::universal::fasc::lazy_expr::ConstantId;
use crate::fixed_point::universal::tier_types::{CompactShadow, ShadowConstantId};
use crate::fixed_point::width::WideInt;

type R<X> = Result<X, MpfErr>;

/// `10^k` exactly (or with the backend's bound).
fn pow10<T: FascFormat, B: Num<T>>(k: u32) -> R<B> {
    let mut v = B::int(1)?;
    let ten = B::int(10)?;
    for _ in 0..k { v = v.mul(ten)?; }
    Ok(v)
}

/// An unsigned 128-bit denominator exactly.
fn from_u128<T: FascFormat, B: Num<T>>(d: u128) -> R<B> {
    B::int((d >> 1) as i128)?.mul(B::int(2)?)?.add(B::int((d & 1) as i128)?)
}

fn rational<T: FascFormat, B: Num<T>>(r: &RationalNumber) -> R<B> {
    match r.extract_native() {
        RationalParts::Tiny(n, d) => B::int(n as i128)?.div(B::int(d as i128)?),
        RationalParts::Small(n, d) => B::int(n as i128)?.div(B::int(d as i128)?),
        RationalParts::Medium(n, d) => B::int(n as i128)?.div(B::int(d as i128)?),
        RationalParts::Large(n, d) => B::int(n as i128)?.div(B::int(d as i128)?),
        RationalParts::Huge(n, d) => B::int(n)?.div(from_u128::<T, B>(d)?),
        #[cfg(not(feature = "embedded"))]
        RationalParts::Massive(n, d) => B::from_word(n, 0)?.div(B::from_word(d, 0)?),
        RationalParts::Ultra(n, d) => B::from_word(n, 0)?.div(B::from_word(d, 0)?),
        #[cfg(feature = "infinite-precision")]
        RationalParts::Infinite(..) => Err(MpfErr::Unsupported),
    }
}

fn constant<T: FascFormat, B: Num<T>>(id: ConstantId) -> R<B> {
    match id {
        ConstantId::Pi => B::pi(),
        ConstantId::E => B::int(1)?.exp(),
        ConstantId::Sqrt2 => B::int(2)?.sqrt(),
        ConstantId::Phi => Ok(B::int(1)?.add(B::int(5)?.sqrt()?)?.half()),
        ConstantId::EulerGamma => B::euler_gamma(),
    }
}

fn shadow_value<T: FascFormat, B: Num<T>>(s: &CompactShadow) -> Option<R<B>> {
    if let CompactShadow::ConstantRef(id) = s {
        return Some(match id {
            ShadowConstantId::Pi => constant::<T, B>(ConstantId::Pi),
            ShadowConstantId::E => constant::<T, B>(ConstantId::E),
            ShadowConstantId::Sqrt2 => constant::<T, B>(ConstantId::Sqrt2),
            ShadowConstantId::Phi => constant::<T, B>(ConstantId::Phi),
            ShadowConstantId::Ln2 => B::int(2).and_then(|v| v.ln()),
            ShadowConstantId::Ln10 => B::int(10).and_then(|v| v.ln()),
            ShadowConstantId::EulerGamma => B::euler_gamma(),
        });
    }
    let (n, d) = s.as_rational()?;
    Some(B::int(n).and_then(|n| n.div(from_u128::<T, B>(d)?)))
}

/// Fraction bits of a binary raw at a UGOD tier (as `StackValue::to_rational`).
pub(super) fn binary_tier_frac<T: FascFormat>(tier: u8) -> u32 {
    match tier { 1 => T::TIER1_RATIONAL_FRAC_BITS, 2 => 32, 3 => 64, 4 => 128, 5 => 256, 6 => 512, _ => 64 }
}

/// The exact value of an evaluated leaf.
fn leaf<T: FascFormat, B: Num<T>>(v: &StackValue<T>) -> R<B> {
    let shadow = match v {
        StackValue::<T>::Binary(_, _, s) | StackValue::<T>::BinaryCompute(_, _, s) | StackValue::<T>::Decimal(_, _, s)
        | StackValue::<T>::DecimalCompute(_, _, s) | StackValue::<T>::Ternary(_, _, s) => Some(s),
        _ => None,
    };
    if let Some(r) = shadow.and_then(shadow_value::<T, B>) { return r; }
    match v {
        StackValue::<T>::Binary(tier, raw, _) => B::from_word(*raw, binary_tier_frac::<T>(*tier)),
        StackValue::<T>::BinaryCompute(_, raw, _) => B::from_word(*raw, 2 * T::FRAC_BITS),
        StackValue::<T>::Decimal(dp, scaled, _) => B::from_word(*scaled, 0)?.div(pow10::<T, B>(*dp as u32)?),
        StackValue::<T>::DecimalCompute(_, raw, _) => B::from_word(*raw, 0)?.div(pow10::<T, B>(T::DECIMAL_COMPUTE_DP as u32)?),
        StackValue::<T>::Ternary(..) | StackValue::<T>::Symbolic(_) => rational::<T, B>(&v.to_rational().map_err(|_| MpfErr::Unsupported)?),
        StackValue::<T>::Error(_) => Err(MpfErr::Unsupported),
    }
}

/// The tree on a backend.
pub(super) fn tree<T: FascFormat, B: Num<T>>(ev: &mut StackEvaluator<T>, e: &LazyExpr<T>) -> R<B> {
    use LazyExpr as L;
    match e {
        L::Literal(_) | L::Value(_) => {
            let v = ev.evaluate(e).map_err(|_| MpfErr::Unsupported)?;
            leaf::<T, B>(&v)
        }
        L::Constant(id) => constant::<T, B>(*id),
        L::Variable(_) => Err(MpfErr::Unsupported),
        L::Negate(a) => Ok(tree::<T, B>(ev, a)?.neg()),
        L::Add(a, b) => tree::<T, B>(ev, a)?.add(tree::<T, B>(ev, b)?),
        L::Sub(a, b) => tree::<T, B>(ev, a)?.sub(tree::<T, B>(ev, b)?),
        L::Mul(a, b) => tree::<T, B>(ev, a)?.mul(tree::<T, B>(ev, b)?),
        L::Div(a, b) => tree::<T, B>(ev, a)?.div(tree::<T, B>(ev, b)?),
        L::Pow(a, b) => tree::<T, B>(ev, a)?.pow(tree::<T, B>(ev, b)?),
        L::Atan2(a, b) => tree::<T, B>(ev, a)?.atan2(tree::<T, B>(ev, b)?),
        L::Exp(a) => tree::<T, B>(ev, a)?.exp(),
        L::Ln(a) => tree::<T, B>(ev, a)?.ln(),
        L::Sqrt(a) => tree::<T, B>(ev, a)?.sqrt(),
        L::Sin(a) => tree::<T, B>(ev, a)?.sin(),
        L::Cos(a) => tree::<T, B>(ev, a)?.cos(),
        L::Tan(a) => tree::<T, B>(ev, a)?.tan(),
        L::Atan(a) => tree::<T, B>(ev, a)?.atan(),
        L::Asin(a) => tree::<T, B>(ev, a)?.asin(),
        L::Acos(a) => tree::<T, B>(ev, a)?.acos(),
        L::Sinh(a) => tree::<T, B>(ev, a)?.sinh(),
        L::Cosh(a) => tree::<T, B>(ev, a)?.cosh(),
        L::Tanh(a) => tree::<T, B>(ev, a)?.tanh(),
        L::Asinh(a) => tree::<T, B>(ev, a)?.asinh(),
        L::Acosh(a) => tree::<T, B>(ev, a)?.acosh(),
        L::Atanh(a) => tree::<T, B>(ev, a)?.atanh(),
    }
}

/// The storage raw of the tree, rounded once (`Err` carries why not): the
/// evaluation runs at the narrowest capacity that holds the format's word
/// with room, and again at the next ones while the rounding is undecided.
pub(super) fn rounded<T: FascFormat>(ev: &mut StackEvaluator<T>, e: &LazyExpr<T>) -> Result<T::Storage, MpfErr> {
    let bits = <T::Storage as WideInt>::BITS;
    // the compute tier first: most trees are decided there
    if let Ok(r) = tree::<T, Fast<T>>(ev, e).and_then(|v: Fast<T>| Num::<T>::round(&v, false)) { return Ok(r); }
    macro_rules! attempt {
        ($C:ty, $tie:expr) => {
            match tree::<T, Mpf<$C>>(ev, e).and_then(|v| Num::<T>::round(&v, $tie)) {
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



/// The decimal reading of `v`: the nearest integer to `v 10^dp`, ties to
/// even, at the most places from `smax` down whose value fits the storage
/// word (as decimal compute values are read).
fn decimal_at<T: FascFormat, B: Num<T>>(v: B, smax: u8, tie: bool) -> R<(u8, T::Storage)> {
    let mut dp = smax as i32;
    while dp >= 0 {
        let y = v.mul(pow10::<T, B>(dp as u32)?)?;
        match y.round_int_even(tie) {
            Ok(r) => return Ok((dp as u8, r)),
            Err(MpfErr::Overflow) => dp -= 1,
            Err(e) => return Err(e),
        }
    }
    Err(MpfErr::Overflow)
}

/// The tree's decimal reading `(places, scaled)`, rounded once from the
/// exact value: the compute tier first, then `Mpf` while undecided.
pub(super) fn rounded_decimal<T: FascFormat>(ev: &mut StackEvaluator<T>, e: &LazyExpr<T>) -> R<(u8, T::Storage)> {
    let smax = T::DECIMAL_STORAGE_MAX_DP;
    if let Ok(r) = tree::<T, Fast<T>>(ev, e).and_then(|v| decimal_at::<T, Fast<T>>(v, smax, false)) { return Ok(r); }
    let bits = <T::Storage as WideInt>::BITS;
    macro_rules! attempt {
        ($C:ty, $tie:expr) => {
            match tree::<T, Mpf<$C>>(ev, e).and_then(|v| decimal_at::<T, Mpf<$C>>(v, smax, $tie)) {
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
