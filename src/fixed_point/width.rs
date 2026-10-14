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
use crate::fixed_point::fixed_num::{Fixed, FixedNum};
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
pub trait WideInt:
    Copy
    + std::fmt::Debug
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
    const ZERO: Self;
    fn zero() -> Self;
    fn one() -> Self;
    /// Sign extension of a native integer (every word is at least 32 bits;
    /// the value must fit, which is the caller's contract on `i32`/`i64`).
    fn from_i64(v: i64) -> Self;
    /// The value of an `i128`, which must fit (panics otherwise).
    fn from_i128(v: i128) -> Self;
    /// The value of an `i128` if it fits.
    fn from_i128_checked(v: i128) -> Option<Self>;
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
    /// Bit `i` of the two's complement representation (`i < BITS`).
    fn bit(self, i: u32) -> bool;
    /// Any of the lowest `n` bits set (`n < BITS`).
    fn low_bits_any(self, n: u32) -> bool;
    /// Significant bits of the magnitude (`BITS` for the minimum).
    fn magnitude_bit_length(self) -> u32;
    /// Truncating division with remainder, as `/` and `%` on the native
    /// integers: the quotient toward zero, the remainder with the dividend's
    /// sign. Panics on a zero divisor.
    fn divmod(self, d: Self) -> (Self, Self);
    /// The integer square root of a non-negative value (the largest k with k * k <= self).
    fn isqrt(self) -> Self;

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
    ($t:ty, $u:ty, $bits:expr) => {
        impl sealed::Sealed for $t {}
        impl WideInt for $t {
            const BITS: u32 = $bits;
            const ZERO: Self = 0;
            #[inline] fn zero() -> Self { 0 }
            #[inline] fn one() -> Self { 1 }
            #[inline] fn from_i64(v: i64) -> Self { v as $t }
            #[inline] fn from_i128(v: i128) -> Self { <$t>::try_from(v).expect("value exceeds the word") }
            #[inline] fn from_i128_checked(v: i128) -> Option<Self> { <$t>::try_from(v).ok() }
            #[inline] fn is_negative(self) -> bool { self < 0 }
            #[inline] fn bit_length(self) -> u32 { $bits - self.leading_zeros() }
            #[inline] fn checked_add_wide(self, rhs: Self) -> Option<Self> { <$t>::checked_add(self, rhs) }
            #[inline] fn checked_sub_wide(self, rhs: Self) -> Option<Self> { <$t>::checked_sub(self, rhs) }
            #[inline] fn checked_neg_wide(self) -> Option<Self> { <$t>::checked_neg(self) }
            #[inline] fn shl_bits(self, n: u32) -> Self { self << n }
            #[inline] fn shr_bits(self, n: u32) -> Self { self >> n }
            #[inline] fn bit(self, i: u32) -> bool { (self >> i) & 1 == 1 }
            #[inline] fn low_bits_any(self, n: u32) -> bool { n > 0 && (self as $u) & (<$u>::MAX >> ($bits - n)) != 0 }
            #[inline] fn magnitude_bit_length(self) -> u32 { $bits - self.unsigned_abs().leading_zeros() }
            #[inline] fn divmod(self, d: Self) -> (Self, Self) { (self / d, self % d) }
            #[inline] fn isqrt(self) -> Self { debug_assert!(self >= 0); (self as $u).isqrt() as $t }
        }
    };
}
native_wide_int!(i32, u32, 32);
native_wide_int!(i64, u64, 64);
native_wide_int!(i128, u128, 128);

macro_rules! word_wide_int {
    ($t:ident, $bits:expr, $shr:ty, $divmod:path) => {
        impl sealed::Sealed for $t {}
        impl WideInt for $t {
            const BITS: u32 = $bits;
            const ZERO: Self = $t::zero();
            #[inline] fn zero() -> Self { $t::zero() }
            #[inline] fn one() -> Self { $t::from_i128(1) }
            #[inline] fn from_i64(v: i64) -> Self { $t::from_i128(v as i128) }
            #[inline] fn from_i128(v: i128) -> Self { $t::from_i128(v) }
            #[inline] fn from_i128_checked(v: i128) -> Option<Self> { Some($t::from_i128(v)) }
            #[inline] fn is_negative(self) -> bool { $t::is_negative(self) }
            #[inline] fn bit_length(self) -> u32 { words_bit_length(&self.words) }
            #[inline] fn checked_add_wide(self, rhs: Self) -> Option<Self> { $t::checked_add(self, rhs) }
            #[inline] fn checked_sub_wide(self, rhs: Self) -> Option<Self> { $t::checked_sub(self, rhs) }
            #[inline] fn checked_neg_wide(self) -> Option<Self> { $t::checked_neg(self) }
            #[inline] fn shl_bits(self, n: u32) -> Self { self << n as usize }
            #[inline] fn shr_bits(self, n: u32) -> Self { self >> n as $shr }
            #[inline] fn bit(self, i: u32) -> bool { (self.words[(i / 64) as usize] >> (i % 64)) & 1 == 1 }
            #[inline]
            fn low_bits_any(self, n: u32) -> bool {
                let (w, b) = ((n / 64) as usize, n % 64);
                self.words[..w].iter().any(|&x| x != 0) || (b > 0 && self.words[w] & ((1u64 << b) - 1) != 0)
            }
            #[inline]
            fn magnitude_bit_length(self) -> u32 {
                // -MIN wraps to MIN, whose words read as the magnitude 2^(BITS-1)
                let m = if self.is_negative() { -self } else { self };
                words_bit_length(&m.words)
            }
            #[inline] fn divmod(self, d: Self) -> (Self, Self) { $divmod(self, d) }
            #[inline]
            fn isqrt(self) -> Self {
                // Newton from 2^ceil(bits / 2), monotone until it stops decreasing
                debug_assert!(!self.is_negative());
                if self == Self::ZERO { return self; }
                let bits = self.bit_length();
                let mut k = Self::one().shl_bits((bits + 1) / 2);
                loop {
                    let next = (k + self.divmod(k).0).shr_bits(1);
                    if next >= k { return k; }
                    k = next;
                }
            }
        }
    };
}
#[inline] fn divmod_by_ops<T: std::ops::Div<Output = T> + std::ops::Rem<Output = T> + Copy>(a: T, d: T) -> (T, T) { (a / d, a % d) }
word_wide_int!(I256, 256, u32, divmod_by_ops);
word_wide_int!(I512, 512, usize, divmod_by_ops);
word_wide_int!(I1024, 1024, usize, divmod_by_ops);
word_wide_int!(I2048, 2048, usize, crate::fixed_point::domains::binary_fixed::i2048::i2048_divmod);

/// A word a fixed-point value is stored in, with the wider words that belong
/// to it.
///
/// `Compute` is twice as wide: it holds the exact product of two storage
/// values and is the tier every multi-step computation keeps its state at.
/// `Orient` is the exact accumulator of the orientation predicates and the
/// fused kernels (it holds the exact product of two compute values).
pub trait Storage: WideInt {
    type Compute: WideInt + crate::fixed_point::domains::decimal_fixed::transcendental::DecimalWord<Narrow = Self>;
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
    /// Sign extension of a compute value to the accumulator (exact).
    fn widen_compute(c: Self::Compute) -> Self::Orient;
    /// The accumulator value as a storage value if it fits, `None` otherwise.
    fn narrow_orient(o: Self::Orient) -> Option<Self>;
    /// The accumulator value as a compute value if it fits, `None` otherwise.
    fn narrow_orient_to_compute(o: Self::Orient) -> Option<Self::Compute>;

    /// The code the serialized format names this word by: 1 for i32, 2 for
    /// i64, 3 for i128, 4 for I256, 5 for I512.
    #[doc(hidden)]
    const CODE: u8;
    /// The word's size in bytes.
    #[doc(hidden)]
    const BYTES: usize;
    /// The word as big-endian bytes (`BYTES` of them).
    #[doc(hidden)]
    fn to_be_vec(self) -> Vec<u8>;
    /// The word from its first `BYTES` big-endian bytes.
    #[doc(hidden)]
    fn from_be_slice(bytes: &[u8]) -> Self;
    /// `sum a_i * b_i` on the compute word by a width-specific fast path,
    /// `None` when there is none or its preconditions fail. The same integer
    /// as the checked loop; the caller takes the checked loop on `None`.
    #[inline]
    fn dot_acc_fast(_a: &[Self], _b: &[Self]) -> Option<Self::Compute> {
        None
    }

    // The fused kernels' word-specific fast paths (hidden): the 32-bit word
    // runs the softmax numerators on i64, the reciprocal root on u128 and
    // the dot of many keys on AVX2; the 32- and 64-bit words gate through
    // the Q64.64 sigmoid and entropy kernels. Each returns `None` (or
    // `false`) where the word has no fast path and the generic body applies.

    #[doc(hidden)]
    fn mix_numerators_fast<'a, const F: u32>(_exps: &[Self::Compute], _row: &dyn Fn(usize) -> &'a [Self], _num: &mut [Self::Compute]) -> bool
    where
        Fixed<Self, F>: FixedNum<Storage = Self>,
    {
        false
    }
    #[doc(hidden)]
    fn sigmoid_mul_fast<const F: u32>(_x: Self, _gate: Self) -> Option<Self>
    where
        Fixed<Self, F>: FixedNum<Storage = Self>,
    {
        None
    }
    #[doc(hidden)]
    fn entropy_fast<const F: u32>(_weights: &[Self]) -> Option<Result<Self, OverflowDetected>>
    where
        Fixed<Self, F>: FixedNum<Storage = Self>,
    {
        None
    }
    #[doc(hidden)]
    fn dot_many_into_fast<const F: u32>(_query: &[Self], _keys_flat: &[Self], _dim: usize, _out: &mut [Self]) -> bool
    where
        Fixed<Self, F>: FixedNum<Storage = Self>,
    {
        false
    }
    #[doc(hidden)]
    fn rms_apply_in_place_fast<const F: u32>(_x: &mut [Self], _w: &[Self], _inv: Self::Compute, _shift: u32) -> Option<Result<(), OverflowDetected>>
    where
        Fixed<Self, F>: FixedNum<Storage = Self>,
    {
        None
    }
    #[doc(hidden)]
    fn rms_reciprocal_fast<const F: u32>(_values: &[Self], _eps_q64: i128) -> Option<Result<(Self::Compute, u32), OverflowDetected>>
    where
        Fixed<Self, F>: FixedNum<Storage = Self>,
    {
        None
    }
}

/// Storage words with a second exact accumulator, wide enough for the circle
/// predicates. The 512-bit word has none (they would need more than 2048
/// bits), so those predicates are absent there by trait bound.
pub trait CircleStorage: Storage {
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
    const CODE: u8 = 1;
    const BYTES: usize = 4;
    #[inline] fn to_be_vec(self) -> Vec<u8> { self.to_be_bytes().to_vec() }
    #[inline] fn from_be_slice(bytes: &[u8]) -> Self { i32::from_be_bytes(bytes[..4].try_into().expect("4 bytes")) }
    type Compute = i64;
    type Orient = i128; // W = 32: 66 / 99 bits
    #[inline] fn widen(self) -> i64 { self as i64 }
    #[inline] fn mul_wide(self, rhs: Self) -> i64 { (self as i64) * (rhs as i64) }
    #[inline] fn narrow(c: i64) -> Option<Self> { i32::try_from(c).ok() }
    #[inline] fn orient(self) -> i128 { self as i128 }
    #[inline] fn widen_product(a: i64, b: i64) -> i128 { (a as i128) * (b as i128) }
    #[inline] fn widen_compute(c: i64) -> i128 { c as i128 }
    #[inline] fn narrow_orient(o: i128) -> Option<Self> { i32::try_from(o).ok() }
    #[inline] fn narrow_orient_to_compute(o: i128) -> Option<i64> { i64::try_from(o).ok() }
    #[cfg(target_arch = "x86_64")]
    #[inline]
    fn dot_acc_fast(a: &[i32], b: &[i32]) -> Option<i64> {
        use crate::fixed_point::imperative::linalg::{long_dot_acc_i32, BOUNDED_DOT_MIN_LEN};
        if a.len() < BOUNDED_DOT_MIN_LEN { None } else { long_dot_acc_i32(a, b) }
    }

    fn mix_numerators_fast<'a, const F: u32>(exps: &[i64], row: &dyn Fn(usize) -> &'a [i32], num: &mut [i64]) -> bool
    where
        Fixed<i32, F>: FixedNum<Storage = i32>,
    {
        if F > 15 || exps.len() > 1usize << (31 - F) {
            return false;
        }
        let round = 1i64 << (F - 1);
        for (j, &e) in exps.iter().enumerate() {
            let v = row(j);
            for (n, &x) in num.iter_mut().zip(v) {
                *n += (e * x as i64 + round) >> F;
            }
        }
        true
    }
    fn sigmoid_mul_fast<const F: u32>(x: i32, gate: i32) -> Option<i32>
    where
        Fixed<i32, F>: FixedNum<Storage = i32>,
    {
        let p = (x as i128) * gate_sigmoid_q64::<F>(gate as i128);
        Some(((p >> 64) + ((p >> 63) & 1)) as i32)
    }
    fn entropy_fast<const F: u32>(weights: &[i32]) -> Option<Result<i32, OverflowDetected>>
    where
        Fixed<i32, F>: FixedNum<Storage = i32>,
    {
        Some(entropy_q64::<F>(weights.iter().map(|&w| w as i128)).and_then(|r| i32::try_from(r).map_err(|_| OverflowDetected::TierOverflow)))
    }
    #[cfg(target_arch = "x86_64")]
    fn dot_many_into_fast<const F: u32>(q: &[i32], keys_flat: &[i32], dim: usize, out: &mut [i32]) -> bool
    where
        Fixed<i32, F>: FixedNum<Storage = i32>,
    {
        use crate::fixed_point::imperative::linalg::{max_abs_avx2, unchecked_dot_avx2};
        if !std::is_x86_feature_detected!("avx2") {
            return false;
        }
        let q_max = unsafe { max_abs_avx2(q) };
        if (dim as u128) * (q_max as u128) * (1u128 << 31) >= 1u128 << 63 {
            return false;
        }
        for (o, key) in out.iter_mut().zip(keys_flat.chunks_exact(dim)) {
            *o = crate::fixed_point::imperative::kernels::round_to_storage::<Fixed<i32, F>>(unsafe { unchecked_dot_avx2(q, key) });
        }
        true
    }
    #[inline]
    fn rms_apply_in_place_fast<const F: u32>(x: &mut [i32], w: &[i32], inv: i64, shift: u32) -> Option<Result<(), OverflowDetected>>
    where
        Fixed<i32, F>: FixedNum<Storage = i32>,
    {
        for (v, &w) in x.iter_mut().zip(w) {
            let product = (*v as i64 * w as i64) as i128 * inv as i128;
            let rounded = ((product >> (shift - 1)) + 1) >> 1;
            match i32::try_from(rounded) {
                Ok(r) => *v = r,
                Err(_) => return Some(Err(OverflowDetected::TierOverflow)),
            }
        }
        Some(Ok(()))
    }
    fn rms_reciprocal_fast<const F: u32>(values: &[i32], eps_q64: i128) -> Option<Result<(i64, u32), OverflowDetected>>
    where
        Fixed<i32, F>: FixedNum<Storage = i32>,
    {
        const OVERFLOW: OverflowDetected = OverflowDetected::TierOverflow;
        let n = values.len() as u128;
        let mut sum_sq = 0u128;
        for &v in values {
            let raw = v as i64;
            match sum_sq.checked_add((raw * raw) as u128) {
                Some(s) => sum_sq = s,
                None => return Some(Err(OVERFLOW)),
            }
        }
        // T = n * (mean + eps) at Q64.64: the sum of squares (2F fraction bits)
        // moved up to 64, plus n * eps
        let up = 64 - 2 * F;
        if sum_sq.leading_zeros() <= up + 1 {
            return Some(Err(OVERFLOW));
        }
        let Some(n_eps) = i128::try_from(n).ok().and_then(|n| n.checked_mul(eps_q64)) else { return Some(Err(OVERFLOW)) };
        let Some(total) = ((sum_sq << up) as i128).checked_add(n_eps) else { return Some(Err(OVERFLOW)) };
        if total < 0 {
            return Some(Err(OverflowDetected::DomainError));
        }
        if total == 0 {
            return Some(Err(OverflowDetected::DivisionByZero));
        }
        // mean + eps = q / 2^(64 + t_shift); both shifts even so the root's
        // exponent is whole
        let total = total as u128;
        let t_shift = total.leading_zeros() & !1;
        let q = (total << t_shift) / n;
        if q == 0 {
            return Some(Err(OVERFLOW));
        }
        let q_shift = q.leading_zeros() & !1;
        let root = (q << q_shift).isqrt(); // [2^63, 2^64): sqrt(mean + eps) * 2^half
        let half = (64 + t_shift + q_shift) / 2;
        let inv = ((1u128 << 125) / root) as i64; // 2^(125 - half) / sqrt(mean + eps)
        // storage raw = x_raw * w_raw / 2^F * inv / 2^(125 - half)
        // half <= 111 for any length below 2^32 (shift >= 16); refuse the rest
        Some((125 + F).checked_sub(half).filter(|&shift| shift >= 1).map(|shift| (inv, shift)).ok_or(OVERFLOW))
    }
}
impl CircleStorage for i32 {
    type Circle = I256; // 133 / 166 bits
    #[inline] fn circle(self) -> I256 { I256::from_i128(self as i128) }
}

impl Storage for i64 {
    const CODE: u8 = 2;
    const BYTES: usize = 8;
    #[inline] fn to_be_vec(self) -> Vec<u8> { self.to_be_bytes().to_vec() }
    #[inline] fn from_be_slice(bytes: &[u8]) -> Self { i64::from_be_bytes(bytes[..8].try_into().expect("8 bytes")) }
    type Compute = i128;
    type Orient = I256; // W = 64: 130 / 195 bits
    #[inline] fn widen(self) -> i128 { self as i128 }
    #[inline] fn mul_wide(self, rhs: Self) -> i128 { (self as i128) * (rhs as i128) }
    #[inline] fn narrow(c: i128) -> Option<Self> { i64::try_from(c).ok() }
    #[inline] fn orient(self) -> I256 { I256::from_i128(self as i128) }
    // sign-correct by itself
    #[inline] fn widen_product(a: i128, b: i128) -> I256 { mul_i128_to_i256(a, b) }
    #[inline] fn widen_compute(c: i128) -> I256 { I256::from_i128(c) }
    #[inline]
    fn narrow_orient(o: I256) -> Option<Self> {
        if o.fits_in_i128() { i64::try_from(o.as_i128()).ok() } else { None }
    }
    #[inline]
    fn narrow_orient_to_compute(o: I256) -> Option<i128> {
        if o.fits_in_i128() { Some(o.as_i128()) } else { None }
    }

    fn sigmoid_mul_fast<const F: u32>(x: i64, gate: i64) -> Option<i64>
    where
        Fixed<i64, F>: FixedNum<Storage = i64>,
    {
        let p = crate::fixed_point::i256::mul_i128_to_i256(x as i128, gate_sigmoid_q64::<F>(gate as i128));
        let mut q = p >> 64u32;
        if p.words[0] >= 1u64 << 63 {
            q = q + crate::fixed_point::I256::from_i128(1);
        }
        Some(q.as_i128() as i64)
    }
    fn entropy_fast<const F: u32>(weights: &[i64]) -> Option<Result<i64, OverflowDetected>>
    where
        Fixed<i64, F>: FixedNum<Storage = i64>,
    {
        Some(entropy_q64::<F>(weights.iter().map(|&w| w as i128)).and_then(|r| i64::try_from(r).map_err(|_| OverflowDetected::TierOverflow)))
    }
}
impl CircleStorage for i64 {
    type Circle = I512; // 261 / 326 bits
    #[inline] fn circle(self) -> I512 { I512::from_i128(self as i128) }
}

impl Storage for i128 {
    const CODE: u8 = 3;
    const BYTES: usize = 16;
    #[inline] fn to_be_vec(self) -> Vec<u8> { self.to_be_bytes().to_vec() }
    #[inline] fn from_be_slice(bytes: &[u8]) -> Self { i128::from_be_bytes(bytes[..16].try_into().expect("16 bytes")) }
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
    #[inline] fn widen_compute(c: I256) -> I512 { I512::from_i256(c) }
    #[inline]
    fn narrow_orient(o: I512) -> Option<Self> {
        if o.fits_in_i128() { Some(o.as_i128()) } else { None }
    }
    #[inline]
    fn narrow_orient_to_compute(o: I512) -> Option<I256> {
        if o.fits_in_i256() { Some(o.as_i256()) } else { None }
    }
}
impl CircleStorage for i128 {
    type Circle = I1024; // 517 / 646 bits
    #[inline] fn circle(self) -> I1024 { I1024::from_i128(self) }
}

impl Storage for I256 {
    const CODE: u8 = 4;
    const BYTES: usize = 32;
    #[inline] fn to_be_vec(self) -> Vec<u8> { self.to_bytes_le().into_iter().rev().collect() }
    #[inline] fn from_be_slice(bytes: &[u8]) -> Self { I256::from_bytes_le(&bytes[..32].iter().rev().copied().collect::<Vec<u8>>()) }
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
    #[inline] fn widen_compute(c: I512) -> I1024 { I1024::from_i512(c) }
    #[inline]
    fn narrow_orient(o: I1024) -> Option<Self> {
        if o.fits_in_i256() { Some(o.as_i256()) } else { None }
    }
    #[inline]
    fn narrow_orient_to_compute(o: I1024) -> Option<I512> {
        if o.fits_in_i512() { Some(o.as_i512()) } else { None }
    }
}
impl CircleStorage for I256 {
    type Circle = I2048; // 1029 / 1286 bits
    #[inline] fn circle(self) -> I2048 { I2048::from_i256(self) }
}

impl Storage for I512 {
    const CODE: u8 = 5;
    const BYTES: usize = 64;
    #[inline] fn to_be_vec(self) -> Vec<u8> { self.to_bytes_le().into_iter().rev().collect() }
    #[inline] fn from_be_slice(bytes: &[u8]) -> Self { I512::from_bytes_le(&bytes[..64].iter().rev().copied().collect::<Vec<u8>>()) }
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
    #[inline] fn widen_compute(c: I1024) -> I2048 { I2048::from_i1024(c) }
    #[inline]
    fn narrow_orient(o: I2048) -> Option<Self> {
        if o.fits_in_i512() { Some(o.as_i512()) } else { None }
    }
    #[inline]
    fn narrow_orient_to_compute(o: I2048) -> Option<I1024> {
        if o.fits_in_i1024() { Some(o.as_i1024()) } else { None }
    }
}

/// sigma(gate) at Q64.64 for a raw of F fraction bits (given as i128),
/// through the wide kernel.
#[inline]
fn gate_sigmoid_q64<const F: u32>(gate: i128) -> i128 {
    crate::fixed_point::wide::sigmoid_q64(gate << (64 - F))
}

/// -sum w ln w at Q64.64 for raws of F fraction bits (given as i128), the
/// result at F fraction bits.
fn entropy_q64<const F: u32>(weights: impl Iterator<Item = i128>) -> Result<i128, OverflowDetected> {
    use crate::fixed_point::wide::{ln_q64, narrow_q64};
    let mut sum: i128 = 0;
    for w in weights {
        let w_q64 = w << (64 - F);
        if let Some(ln) = ln_q64(w_q64) {
            let p = crate::fixed_point::i256::mul_i128_to_i256(w_q64, ln);
            let mut term = p >> 64u32;
            if p.words[0] >= 1u64 << 63 {
                term = term + crate::fixed_point::I256::from_i128(1);
            }
            if !term.fits_in_i128() {
                return Err(OverflowDetected::TierOverflow);
            }
            sum = sum.checked_add(term.as_i128()).ok_or(OverflowDetected::TierOverflow)?;
        }
    }
    Ok(narrow_q64(sum.checked_neg().ok_or(OverflowDetected::TierOverflow)?, F))
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
        assert!(v(5).bit(0) && v(5).bit(2) && !v(5).bit(1));
        assert!(v(-1).bit(T::BITS - 1) && !v(1).bit(T::BITS - 1));
        assert!(v(8).low_bits_any(4) && !v(8).low_bits_any(3) && v(-8).low_bits_any(4) && !v(0).low_bits_any(T::BITS - 1));
        assert_eq!(v(-255).magnitude_bit_length(), 8);
        assert_eq!(v(0).magnitude_bit_length(), 0);
        for &(a, d) in &[(7i64, 2i64), (-7, 2), (7, -2), (-7, -2), (0, 3), (6, 3), (-6, -3), (1, 1 << 20)] {
            assert_eq!(v(a).divmod(v(d)), (v(a / d), v(a % d)), "{a} / {d}");
        }

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
        assert_eq!(min.magnitude_bit_length(), T::BITS);
        assert_eq!(max.magnitude_bit_length(), T::BITS - 1);
        let (q, r) = max.divmod(v(7));
        assert_eq!(q.shl_bits(2) + q.shl_bits(1) + q + r, max); // 7q + r, without a product near the budget
        assert!(!r.is_negative() && r < v(7));
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
        // widening a compute value and narrowing back are exact and checked
        assert_eq!(S::widen_compute(-big), -o(1).shl_bits(2 * S::BITS - 2));
        assert_eq!(S::narrow_orient_to_compute(S::widen_compute(-big)), Some(-big));
        assert_eq!(S::narrow_orient_to_compute(o(1).shl_bits(2 * S::BITS - 1)), None);
        assert_eq!(S::narrow_orient_to_compute(-o(1).shl_bits(2 * S::BITS - 1)), Some(c(1).shl_bits(2 * S::BITS - 1)));
        assert_eq!(S::narrow_orient(o(-77)), Some(s(-77)));
        assert_eq!(S::narrow_orient(o(1).shl_bits(S::BITS - 1)), None);
        assert_eq!(S::narrow_orient(-o(1).shl_bits(S::BITS - 1)), Some(min));
        assert_eq!(S::narrow_orient(-o(1).shl_bits(S::BITS - 1) - o(1)), None);
    }

    #[test]
    fn every_storage_word_widens_and_narrows_exactly() {
        check_storage::<i32>();
        check_storage::<i64>();
        check_storage::<i128>();
        check_storage::<I256>();
        check_storage::<I512>();
    }

    /// 2^1023 does not fit an I1024 (it is one above the maximum); before
    /// 0.6.8 `I2048::fits_in_i1024` read the sign from the wrong word and
    /// accepted every value in [2^1023, 2^1024).
    #[test]
    fn the_fits_checks_refuse_the_value_one_above_the_maximum() {
        assert!(!I2048::from_i64(1).shl_bits(1023).fits_in_i1024());
        assert!(I2048::from_i64(1).shl_bits(1022).fits_in_i1024());
        assert!((-I2048::from_i64(1).shl_bits(1023)).fits_in_i1024());
        assert!(!I1024::from_i64(1).shl_bits(511).fits_in_i512());
        assert!(!I512::from_i64(1).shl_bits(255).fits_in_i256());
        assert!(!I256::from_i64(1).shl_bits(127).fits_in_i128());
    }

    #[test]
    fn circle_accumulators_sign_extend() {
        assert_eq!((-3i32).circle(), I256::from_i128(-3));
        assert_eq!((-3i64).circle(), I512::from_i128(-3));
        assert_eq!((-3i128).circle(), I1024::from_i128(-3));
        assert_eq!(I256::from_i128(-3).circle(), I2048::from_i128(-3));
    }
}
