//! Width traits: what an integer word can do, and which wider words belong
//! to a storage word.
//!
//! [`WideInt`] is one uniform surface over every integer the crate computes
//! on (`i32`, `i64`, `i128`, `I256`, `I512`, `I1024`, `I2048`). [`Storage`]
//! is implemented for the five words a fixed-point value can be stored in and
//! names, per word, the compute word (twice as wide) and the exact
//! accumulator used by the predicates and the fused kernels. [`CircleStorage`]
//! adds the second, wider accumulator for the storage words that have one.
//!
//! The associated types depend on the storage word only, never on the number
//! of fractional bits. Both traits are sealed.
//!
//! Design: `docs/design/WIDTH_AS_TYPE.md`.

// The traits are complete ahead of their users: the generic `Fixed<S, F>`
// consumes the rest of the surface as the migration proceeds.
#![allow(dead_code)]

use crate::fixed_point::core_types::errors::OverflowDetected;
use crate::fixed_point::i256::mul_i128_to_i256;
use crate::fixed_point::{I1024, I2048, I256, I512};

mod sealed {
    pub trait Sealed {}
}

/// Exact signed integer arithmetic on a fixed-width word.
///
/// Products are formed on magnitudes and the sign reapplied, so the wide
/// types' `Mul` implementations are only ever entered with non-negative
/// operands (a schoolbook or a truncated unsigned product, both exact there).
/// Each product first asserts that the operands' bit lengths fit the type,
/// which is the width budget made loud. Sums are checked: a running sum that
/// leaves the word is a typed `TierOverflow`, never a wrap.
pub(crate) trait WideInt:
    Copy
    + Eq
    + Ord
    + Send
    + Sync
    + 'static
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Neg<Output = Self>
    + sealed::Sealed
{
    const BITS: u32;
    fn zero() -> Self;
    fn one() -> Self;
    /// Sign extension of a native integer (every word is at least 32 bits;
    /// the value must fit, which is the caller's contract on `i32`/`i64`).
    fn from_i64(v: i64) -> Self;
    fn is_negative(self) -> bool;
    /// Significant bits of a NON-NEGATIVE value.
    fn bit_length(self) -> u32;
    /// Checked signed addition (`None` on overflow).
    fn checked_add_wide(self, rhs: Self) -> Option<Self>;
    /// Checked signed subtraction (`None` on overflow).
    fn checked_sub_wide(self, rhs: Self) -> Option<Self>;
    /// Checked negation (`None` for the most negative value).
    fn checked_neg_wide(self) -> Option<Self>;
    /// Left shift by `n < BITS` bits (bits shifted out are lost).
    fn shl_bits(self, n: u32) -> Self;
    /// Arithmetic right shift by `n < BITS` bits (floor).
    fn shr_bits(self, n: u32) -> Self;

    #[inline]
    fn mul_exact(self, rhs: Self) -> Self {
        let neg = self.is_negative() != rhs.is_negative();
        let a = if self.is_negative() { -self } else { self };
        let b = if rhs.is_negative() { -rhs } else { rhs };
        assert!(
            a.bit_length() + b.bit_length() <= Self::BITS - 1,
            "exact predicate: accumulator width budget violated"
        );
        let p = a * b;
        if neg { -p } else { p }
    }

    #[inline]
    fn add_exact(self, rhs: Self) -> Result<Self, OverflowDetected> {
        self.checked_add_wide(rhs).ok_or(OverflowDetected::TierOverflow)
    }
}

#[inline]
pub(crate) fn words_bit_length(words: &[u64]) -> u32 {
    for i in (0..words.len()).rev() {
        if words[i] != 0 {
            return i as u32 * 64 + (64 - words[i].leading_zeros());
        }
    }
    0
}

macro_rules! native_wide_int {
    ($t:ty, $bits:expr) => {
        impl sealed::Sealed for $t {}
        impl WideInt for $t {
            const BITS: u32 = $bits;
            #[inline] fn zero() -> Self { 0 }
            #[inline] fn one() -> Self { 1 }
            #[inline] fn from_i64(v: i64) -> Self { v as $t }
            #[inline] fn is_negative(self) -> bool { self < 0 }
            #[inline] fn bit_length(self) -> u32 { $bits - self.leading_zeros() }
            #[inline] fn checked_add_wide(self, rhs: Self) -> Option<Self> { <$t>::checked_add(self, rhs) }
            #[inline] fn checked_sub_wide(self, rhs: Self) -> Option<Self> { <$t>::checked_sub(self, rhs) }
            #[inline] fn checked_neg_wide(self) -> Option<Self> { <$t>::checked_neg(self) }
            #[inline] fn shl_bits(self, n: u32) -> Self { self << n }
            #[inline] fn shr_bits(self, n: u32) -> Self { self >> n }
        }
    };
}
native_wide_int!(i32, 32);
native_wide_int!(i64, 64);
native_wide_int!(i128, 128);

macro_rules! word_wide_int {
    ($t:ident, $bits:expr, $shr:ty) => {
        impl sealed::Sealed for $t {}
        impl WideInt for $t {
            const BITS: u32 = $bits;
            #[inline] fn zero() -> Self { $t::zero() }
            #[inline] fn one() -> Self { $t::from_i128(1) }
            #[inline] fn from_i64(v: i64) -> Self { $t::from_i128(v as i128) }
            #[inline] fn is_negative(self) -> bool { $t::is_negative(self) }
            #[inline] fn bit_length(self) -> u32 { words_bit_length(&self.words) }
            #[inline] fn checked_add_wide(self, rhs: Self) -> Option<Self> { $t::checked_add(self, rhs) }
            #[inline] fn checked_sub_wide(self, rhs: Self) -> Option<Self> { $t::checked_sub(self, rhs) }
            #[inline] fn checked_neg_wide(self) -> Option<Self> { $t::checked_neg(self) }
            #[inline] fn shl_bits(self, n: u32) -> Self { self << n as usize }
            #[inline] fn shr_bits(self, n: u32) -> Self { self >> n as $shr }
        }
    };
}
word_wide_int!(I256, 256, u32);
word_wide_int!(I512, 512, usize);
word_wide_int!(I1024, 1024, usize);
word_wide_int!(I2048, 2048, usize);

/// A word a fixed-point value is stored in, with the wider words that belong
/// to it.
///
/// `Compute` is twice as wide: it holds the exact product of two storage
/// values and is the tier every multi-step computation keeps its state at.
/// `Orient` is the exact accumulator of the orientation predicates and the
/// fused kernels (it holds the exact product of two compute values).
pub(crate) trait Storage: WideInt {
    type Compute: WideInt;
    type Orient: WideInt;

    /// Sign extension to the compute word (no shift; exact).
    fn widen(self) -> Self::Compute;
    /// Exact product of two storage values on the compute word.
    fn mul_wide(self, rhs: Self) -> Self::Compute;
    /// The compute value as a storage value if it fits, `None` otherwise.
    fn narrow(c: Self::Compute) -> Option<Self>;
    /// Sign extension to the orientation accumulator (exact).
    fn orient(self) -> Self::Orient;
    /// Exact product of two compute values on the orientation accumulator.
    /// Panics if the operands' bit lengths exceed the accumulator (the width
    /// budget made loud).
    fn widen_product(a: Self::Compute, b: Self::Compute) -> Self::Orient;
}

/// Storage words with a second exact accumulator, wide enough for the circle
/// predicates. The 512-bit word has none (they would need more than 2048
/// bits), so those predicates are absent there by trait bound.
pub(crate) trait CircleStorage: Storage {
    type Circle: WideInt;
    /// Sign extension to the circle accumulator (exact).
    fn circle(self) -> Self::Circle;
}

/// Exact product of two wide values through an unsigned widening multiply:
/// the `mul_to_*` family is UNSIGNED by convention, so it is applied to
/// magnitudes and the sign reapplied.
macro_rules! signed_widening_product {
    ($a:expr, $b:expr, $mul:ident) => {{
        let (a, b) = ($a, $b);
        let a_neg = a.is_negative();
        let b_neg = b.is_negative();
        let abs_a = if a_neg { -a } else { a };
        let abs_b = if b_neg { -b } else { b };
        let product = abs_a.$mul(abs_b);
        if a_neg != b_neg { -product } else { product }
    }};
}

/// As [`signed_widening_product`], with the accumulator budget asserted.
macro_rules! budgeted_widening_product {
    ($a:expr, $b:expr, $mul:ident, $acc:ty) => {{
        let (a, b) = ($a, $b);
        let a_neg = a.is_negative();
        let b_neg = b.is_negative();
        let abs_a = if a_neg { -a } else { a };
        let abs_b = if b_neg { -b } else { b };
        assert!(
            words_bit_length(&abs_a.words) + words_bit_length(&abs_b.words) <= <$acc as WideInt>::BITS - 1,
            "exact predicate: accumulator width budget violated"
        );
        let p = abs_a.$mul(abs_b);
        if a_neg != b_neg { -p } else { p }
    }};
}

impl Storage for i32 {
    type Compute = i64;
    type Orient = i128; // W = 32: 66 / 99 bits
    #[inline] fn widen(self) -> i64 { self as i64 }
    #[inline] fn mul_wide(self, rhs: Self) -> i64 { (self as i64) * (rhs as i64) }
    #[inline] fn narrow(c: i64) -> Option<Self> { i32::try_from(c).ok() }
    #[inline] fn orient(self) -> i128 { self as i128 }
    #[inline] fn widen_product(a: i64, b: i64) -> i128 { (a as i128) * (b as i128) }
}
impl CircleStorage for i32 {
    type Circle = I256; // 133 / 166 bits
    #[inline] fn circle(self) -> I256 { I256::from_i128(self as i128) }
}

impl Storage for i64 {
    type Compute = i128;
    type Orient = I256; // W = 64: 130 / 195 bits
    #[inline] fn widen(self) -> i128 { self as i128 }
    #[inline] fn mul_wide(self, rhs: Self) -> i128 { (self as i128) * (rhs as i128) }
    #[inline] fn narrow(c: i128) -> Option<Self> { i64::try_from(c).ok() }
    #[inline] fn orient(self) -> I256 { I256::from_i128(self as i128) }
    // sign-correct by itself
    #[inline] fn widen_product(a: i128, b: i128) -> I256 { mul_i128_to_i256(a, b) }
}
impl CircleStorage for i64 {
    type Circle = I512; // 261 / 326 bits
    #[inline] fn circle(self) -> I512 { I512::from_i128(self as i128) }
}

impl Storage for i128 {
    type Compute = I256;
    type Orient = I512; // W = 128: 258 / 387 bits
    #[inline] fn widen(self) -> I256 { I256::from_i128(self) }
    #[inline] fn mul_wide(self, rhs: Self) -> I256 { mul_i128_to_i256(self, rhs) }
    #[inline]
    fn narrow(c: I256) -> Option<Self> {
        if c.fits_in_i128() { Some(c.as_i128()) } else { None }
    }
    #[inline] fn orient(self) -> I512 { I512::from_i128(self) }
    #[inline]
    fn widen_product(a: I256, b: I256) -> I512 {
        budgeted_widening_product!(a, b, mul_to_i512, I512)
    }
}
impl CircleStorage for i128 {
    type Circle = I1024; // 517 / 646 bits
    #[inline] fn circle(self) -> I1024 { I1024::from_i128(self) }
}

impl Storage for I256 {
    type Compute = I512;
    type Orient = I1024; // W = 256: 514 / 771 bits
    #[inline] fn widen(self) -> I512 { I512::from_i256(self) }
    #[inline] fn mul_wide(self, rhs: Self) -> I512 { signed_widening_product!(self, rhs, mul_to_i512) }
    #[inline]
    fn narrow(c: I512) -> Option<Self> {
        if c.fits_in_i256() { Some(c.as_i256()) } else { None }
    }
    #[inline] fn orient(self) -> I1024 { I1024::from_i256(self) }
    #[inline]
    fn widen_product(a: I512, b: I512) -> I1024 {
        budgeted_widening_product!(a, b, mul_to_i1024, I1024)
    }
}
impl CircleStorage for I256 {
    type Circle = I2048; // 1029 / 1286 bits
    #[inline] fn circle(self) -> I2048 { I2048::from_i256(self) }
}

impl Storage for I512 {
    type Compute = I1024;
    type Orient = I2048; // W = 512: 1026 / 1539 bits; circle predicates exceed 2048
    #[inline] fn widen(self) -> I1024 { I1024::from_i512(self) }
    #[inline] fn mul_wide(self, rhs: Self) -> I1024 { signed_widening_product!(self, rhs, mul_to_i1024) }
    #[inline]
    fn narrow(c: I1024) -> Option<Self> {
        if c.fits_in_i512() { Some(c.as_i512()) } else { None }
    }
    #[inline] fn orient(self) -> I2048 { I2048::from_i512(self) }
    #[inline]
    fn widen_product(a: I1024, b: I1024) -> I2048 {
        budgeted_widening_product!(a, b, mul_to_i2048, I2048)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every identity below is exact integer arithmetic, checked against the
    /// native `i128` on values small enough to fit it.
    fn check_word<T: WideInt + std::fmt::Debug>() {
        let v = |x: i64| T::from_i64(x);
        assert_eq!(T::zero(), v(0));
        assert_eq!(T::one(), v(1));
        assert!(v(-1).is_negative() && !v(0).is_negative() && !v(1).is_negative());
        assert_eq!(v(0).bit_length(), 0);
        assert_eq!(v(1).bit_length(), 1);
        assert_eq!(v(255).bit_length(), 8);
        assert_eq!(v(1).shl_bits(T::BITS - 2).bit_length(), T::BITS - 1);
        assert_eq!(v(1).shl_bits(T::BITS - 2).shr_bits(T::BITS - 2), v(1));
        assert_eq!(v(-5).shr_bits(1), v(-3)); // arithmetic: floor
        assert_eq!(v(-7).shl_bits(3), v(-56));
        assert_eq!(v(1000).checked_add_wide(v(-1001)), Some(v(-1)));
        assert_eq!(v(1000).checked_sub_wide(v(1001)), Some(v(-1)));
        assert_eq!(v(9).checked_neg_wide(), Some(v(-9)));
        assert_eq!(v(-1234).mul_exact(v(567)), v(-1234 * 567));
        assert_eq!(v(-1234).mul_exact(v(-567)), v(1234 * 567));

        // The extremes: max = 2^(BITS-1) - 1, min = -max - 1.
        let max = v(1).shl_bits(T::BITS - 2) - v(1) + v(1).shl_bits(T::BITS - 2);
        let min = -max - v(1);
        assert!(min.is_negative() && !max.is_negative());
        assert_eq!(max.bit_length(), T::BITS - 1);
        assert_eq!(max.checked_add_wide(v(1)), None);
        assert_eq!(min.checked_sub_wide(v(1)), None);
        assert_eq!(min.checked_add_wide(v(-1)), None);
        assert_eq!(max.checked_sub_wide(v(-1)), None);
        assert_eq!(min.checked_neg_wide(), None);
        assert_eq!(max.checked_neg_wide(), Some(min + v(1)));
        assert_eq!(max.checked_add_wide(min), Some(v(-1)));
        assert_eq!(v(0).checked_sub_wide(min), None);
        assert!(max.add_exact(v(1)).is_err());
    }

    #[test]
    fn every_word_has_the_same_integer_semantics() {
        check_word::<i32>();
        check_word::<i64>();
        check_word::<i128>();
        check_word::<I256>();
        check_word::<I512>();
        check_word::<I1024>();
        check_word::<I2048>();
    }

    fn check_storage<S: Storage + std::fmt::Debug>()
    where
        S::Compute: std::fmt::Debug,
        S::Orient: std::fmt::Debug,
    {
        let s = |x: i64| S::from_i64(x);
        let c = |x: i64| S::Compute::from_i64(x);
        let o = |x: i64| S::Orient::from_i64(x);
        for &(a, b) in &[(0i64, 0i64), (1, -1), (-46341, 46340), (30000, -30000), (-7, -9)] {
            assert_eq!(s(a).widen(), c(a));
            assert_eq!(s(a).orient(), o(a));
            assert_eq!(s(a).mul_wide(s(b)), c(a * b));
            assert_eq!(S::widen_product(c(a), c(b)), o(a * b));
            assert_eq!(S::narrow(c(a)), Some(s(a)));
        }
        // Narrowing is checked at the storage extremes: 2^(BITS-1) does not
        // fit, -2^(BITS-1) does.
        let top = c(1).shl_bits(S::BITS - 1);
        assert_eq!(S::narrow(top), None);
        assert_eq!(S::narrow(top - c(1)).map(|m| m.widen()), Some(top - c(1)));
        assert_eq!(S::narrow(-top).map(|m| m.widen()), Some(-top));
        assert_eq!(S::narrow(-top - c(1)), None);
        // The widening product of the two most negative storage values is
        // 2^(2 BITS - 2): exact on the compute word.
        let min = S::narrow(-top).unwrap();
        assert_eq!(min.mul_wide(min), c(1).shl_bits(2 * S::BITS - 2));
        let max = S::narrow(top - c(1)).unwrap();
        assert_eq!(max.mul_wide(min), -(top - c(1)).mul_exact(top));
        // And of two compute values of that size on the accumulator.
        let big = c(1).shl_bits(2 * S::BITS - 2);
        assert_eq!(S::widen_product(big, -big), -o(1).shl_bits(4 * S::BITS - 4));
    }

    #[test]
    fn every_storage_word_widens_and_narrows_exactly() {
        check_storage::<i32>();
        check_storage::<i64>();
        check_storage::<i128>();
        check_storage::<I256>();
        check_storage::<I512>();
    }

    #[test]
    fn circle_accumulators_sign_extend() {
        assert_eq!((-3i32).circle(), I256::from_i128(-3));
        assert_eq!((-3i64).circle(), I512::from_i128(-3));
        assert_eq!((-3i128).circle(), I1024::from_i128(-3));
        assert_eq!(I256::from_i128(-3).circle(), I2048::from_i128(-3));
    }
}
