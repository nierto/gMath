//! `Fixed<S, F>`: a fixed-point format as a type, and the per-width kernels.
//!
//! A format is a storage word `S` and a number of fractional bits `F`.
//! [`FixedNum`] is implemented once per storage word: for `Fixed<i32, F>` at
//! every admitted `F`, and for the four wider words at their one split. Its
//! methods are the kernels that differ by width: the conversions between the
//! storage and compute tiers, the rounded multiply and divide, compute-tier
//! arithmetic, and the routing of each transcendental to the engine of its
//! compute format. Everything above them is written once against the trait.
//!
//! Every body here is the code the matching build profile compiled before
//! width became a type, moved without change, so each width returns the
//! integers it always did. All five impls compile in every build.
//!
//! Design: `docs/design/WIDTH_AS_TYPE.md`.

// The kernels are complete ahead of their users: the generic layers adopt
// them as the migration proceeds.
#![allow(dead_code)]

use crate::fixed_point::core_types::errors::OverflowDetected;
use crate::fixed_point::domains::binary_fixed::transcendental::atan_tier_n_plus_1::{
    atan2_q128_128_compute, atan2_q256_256_compute, atan2_q512_512_native, atan2_q64_64_native,
    atan_q128_128_compute, atan_q128_128_native, atan_q256_256_compute, atan_q512_512_native,
    atan_q64_64_native,
};
use crate::fixed_point::domains::binary_fixed::transcendental::exp_tier_n_plus_1::{
    exp_q128_128_native, exp_q256_256_native, exp_q512_512_native, exp_q64_64_native,
};
use crate::fixed_point::domains::binary_fixed::transcendental::ln_tier_n_plus_1::{
    ln_q128_128_native, ln_q256_256_native, ln_q512_512_native, ln_q64_64_native,
};
use crate::fixed_point::domains::binary_fixed::transcendental::sin_cos_tier_n_plus_1::{
    sincos_q128_128_native, sincos_q256_256_native, sincos_q512_512_native, sincos_q64_64_native,
    PI_HALF_Q128, PI_HALF_Q256, PI_HALF_Q512, PI_HALF_Q64,
};
use crate::fixed_point::domains::binary_fixed::transcendental::sqrt_tier_n_plus_1::{
    sqrt_q128_128_native, sqrt_q256_256_native, sqrt_q512_512_native, sqrt_q64_64_native,
};
use crate::fixed_point::width::{Storage, WideInt};
use crate::fixed_point::{I1024, I2048, I256, I512};

mod sealed {
    pub trait Sealed {}
}

/// A fixed-point number: the raw integer `raw` stands for `raw / 2^F`.
///
/// **Layout guarantee:** `Fixed` is `#[repr(transparent)]` over its raw
/// storage integer, so a value and its raw integer have the same size,
/// alignment and bit pattern, and every bit pattern of the raw type is a
/// valid value.
///
/// The supported formats are `Fixed<i32, F>` for `2 <= F <= 30`,
/// `Fixed<i64, 32>`, `Fixed<i128, 64>`, `Fixed<I256, 128>` and
/// `Fixed<I512, 256>`. Any other combination has no arithmetic: the methods
/// do not exist for it, and a 32-bit split outside the range fails to
/// compile.
///
/// ```compile_fail
/// use g_math::fixed_point::Fixed;
/// let x = Fixed::<i32, 31>::one(); // error: F must be between 2 and 30
/// ```
///
/// ```compile_fail
/// use g_math::fixed_point::Fixed;
/// let x = Fixed::<i128, 40>::one(); // error: no such format
/// ```
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct Fixed<S, const F: u32> {
    pub(crate) raw: S,
}

/// Q16.16 on a 32-bit word. Any `Fixed<i32, F>` with `2 <= F <= 30` is a
/// format of this width.
pub type Q16_16 = Fixed<i32, 16>;
/// Q32.32 on a 64-bit word.
pub type Q32_32 = Fixed<i64, 32>;
/// Q64.64 on a 128-bit word.
pub type Q64_64 = Fixed<i128, 64>;
/// Q128.128 on a 256-bit word.
pub type Q128_128 = Fixed<I256, 128>;
/// Q256.256 on a 512-bit word.
pub type Q256_256 = Fixed<I512, 256>;

/// The compute word of a format: twice the storage width, `2F` fractional
/// bits.
pub type Compute<T> = <<T as FixedNum>::Storage as Storage>::Compute;

/// The kernels of one fixed-point format.
///
/// Compute-tier values are raw integers with `2 * FRAC_BITS` fractional
/// bits. Two routings exist for the transcendental engines because the
/// crate has always had two: `direct_*` is what the imperative methods
/// call, `*_at_compute_tier` what the canonical evaluator calls. They
/// name the same engine everywhere except atan at Q64.64 (see
/// `transcendental/mod.rs`), and each is kept as it was.
pub trait FixedNum: Copy + sealed::Sealed {
    type Storage: Storage;
    /// The value with this raw storage word (`Fixed::from_raw` for generic code).
    fn from_raw_storage(raw: Self::Storage) -> Self;
    /// The raw storage word.
    fn raw_storage(self) -> Self::Storage;
    /// Fractional bits of a storage value.
    const FRAC_BITS: u32;
    /// The storage word's tier in the canonical layer's binary ladder.
    const STORAGE_TIER: u8;
    /// The shift of a magnitude that gives the decompositions' convergence
    /// threshold (`linalg::convergence_threshold`): half the storage word's
    /// fractional bits at its standard split, fixed per word (8 on the
    /// 32-bit word at every split, as calibrated in 0.6.2).
    const CONVERGENCE_SHIFT: u32;

    // ---- between the storage and compute tiers -------------------------
    /// A storage value at the compute tier (exact).
    fn upscale_to_compute(val: Self::Storage) -> Compute<Self>;
    /// Nearest storage value, ties toward +infinity; `TierOverflow` beyond storage.
    fn downscale_to_storage(val: Compute<Self>) -> Result<Self::Storage, OverflowDetected>;
    /// Largest storage value not above `val`.
    fn downscale_to_storage_floor(val: Compute<Self>) -> Result<Self::Storage, OverflowDetected>;
    /// Smallest storage value not below `val`.
    fn downscale_to_storage_ceil(val: Compute<Self>) -> Result<Self::Storage, OverflowDetected>;

    // ---- storage tier ---------------------------------------------------
    /// 1.0 as a storage raw.
    fn one_storage() -> Self::Storage;
    /// The integer `v` as a storage raw; `TierOverflow` outside the range.
    fn try_from_int(v: i32) -> Result<Self::Storage, OverflowDetected>;
    /// The integer part (floor); `TierOverflow` outside the i32 range.
    fn try_to_int(raw: Self::Storage) -> Result<i32, OverflowDetected>;
    /// IEEE 754 bits of the value in a format with `fraction_bits` stored
    /// fraction bits and `exponent_bits` exponent bits (nearest, ties to even).
    fn float_bits(raw: Self::Storage, fraction_bits: u32, exponent_bits: u32) -> u64;
    /// `mantissa x 2^shift` truncated toward zero, sign applied, range-checked.
    fn truncated_raw(mantissa: u64, shift: i32, negative: bool) -> Result<Self::Storage, OverflowDetected>;
    /// The exact product rounded to nearest (ties toward +infinity), range-checked.
    fn try_fixed_multiply(a: Self::Storage, b: Self::Storage) -> Result<Self::Storage, OverflowDetected>;
    /// The exact quotient rounded to nearest (ties toward +infinity), range-checked.
    fn try_fixed_divide(a: Self::Storage, b: Self::Storage) -> Result<Self::Storage, OverflowDetected>;

    // ---- compute tier arithmetic ---------------------------------------
    /// The compute word's maximum.
    fn compute_ceiling() -> Compute<Self>;
    /// True when an exp engine result sits at its overflow sentinel.
    fn exp_sentinel_reached(v: &Compute<Self>) -> bool;
    fn compute_checked_multiply(a: Compute<Self>, b: Compute<Self>) -> Result<Compute<Self>, OverflowDetected>;
    fn compute_checked_divide(a: Compute<Self>, b: Compute<Self>) -> Result<Compute<Self>, OverflowDetected>;
    /// The integer `n` as a compute value.
    fn make_compute_int(n: i64) -> Compute<Self>;
    /// `c * num / den` for integers, `den > 0`, one rounding.
    fn compute_mul_div_int(c: Compute<Self>, num: i64, den: i64) -> Result<Compute<Self>, OverflowDetected>;
    /// `c / n` for a count `n > 0`, one rounding.
    fn compute_div_count(c: Compute<Self>, n: usize) -> Result<Compute<Self>, OverflowDetected>;

    /// `a + b`, panics beyond the compute tier.
    #[inline]
    fn compute_add(a: Compute<Self>, b: Compute<Self>) -> Compute<Self> {
        a.checked_add_wide(b).expect("compute tier: addition overflow")
    }
    #[inline]
    fn compute_checked_add(a: Compute<Self>, b: Compute<Self>) -> Result<Compute<Self>, OverflowDetected> {
        a.checked_add_wide(b).ok_or(OverflowDetected::TierOverflow)
    }
    /// `a - b`, panics beyond the compute tier.
    #[inline]
    fn compute_subtract(a: Compute<Self>, b: Compute<Self>) -> Compute<Self> {
        a.checked_sub_wide(b).expect("compute tier: subtraction overflow")
    }
    #[inline]
    fn compute_checked_subtract(a: Compute<Self>, b: Compute<Self>) -> Result<Compute<Self>, OverflowDetected> {
        Self::compute_checked_add(a, Self::compute_checked_negate(b)?)
    }
    /// `-a`, panics for the compute word's minimum.
    #[inline]
    fn compute_negate(a: Compute<Self>) -> Compute<Self> {
        a.checked_neg_wide().expect("compute tier: negation overflow")
    }
    #[inline]
    fn compute_checked_negate(a: Compute<Self>) -> Result<Compute<Self>, OverflowDetected> {
        a.checked_neg_wide().ok_or(OverflowDetected::TierOverflow)
    }
    /// `a / 2`, floor.
    #[inline]
    fn compute_halve(a: Compute<Self>) -> Compute<Self> {
        a.shr_bits(1)
    }
    #[inline]
    fn compute_is_zero(a: &Compute<Self>) -> bool {
        *a == <Compute<Self> as WideInt>::zero()
    }
    #[inline]
    fn compute_is_negative(a: &Compute<Self>) -> bool {
        a.is_negative()
    }
    #[inline]
    fn compute_multiply(a: Compute<Self>, b: Compute<Self>) -> Compute<Self> {
        Self::compute_checked_multiply(a, b).expect("compute tier: multiplication overflow")
    }
    #[inline]
    fn compute_divide(a: Compute<Self>, b: Compute<Self>) -> Result<Compute<Self>, OverflowDetected> {
        Self::compute_checked_divide(a, b)
    }

    // ---- engines as the imperative methods route them -------------------
    fn direct_exp(x: Compute<Self>) -> Compute<Self>;
    fn direct_ln(x: Compute<Self>) -> Compute<Self>;
    fn direct_sqrt(x: Compute<Self>) -> Compute<Self>;
    fn direct_sin(x: Compute<Self>) -> Compute<Self>;
    fn direct_cos(x: Compute<Self>) -> Compute<Self>;
    fn direct_atan(x: Compute<Self>) -> Compute<Self>;
    fn direct_atan2(y: Compute<Self>, x: Compute<Self>) -> Compute<Self>;
    /// pi/2 as the imperative acos and asin use it.
    fn compute_pi_half() -> Compute<Self>;

    // ---- engines as the canonical evaluator routes them -----------------
    fn exp_at_compute_tier(x: Compute<Self>) -> Compute<Self>;
    fn ln_at_compute_tier(x: Compute<Self>) -> Compute<Self>;
    fn sqrt_at_compute_tier(x: Compute<Self>) -> Compute<Self>;
    fn sin_at_compute_tier(x: Compute<Self>) -> Compute<Self>;
    fn cos_at_compute_tier(x: Compute<Self>) -> Compute<Self>;
    fn sincos_at_compute_tier(x: Compute<Self>) -> (Compute<Self>, Compute<Self>);
    fn atan_at_compute_tier(x: Compute<Self>) -> Compute<Self>;
    fn atan2_at_compute_tier(y: Compute<Self>, x: Compute<Self>) -> Compute<Self>;
    /// pi/2 at the compute tier's full precision.
    fn pi_half_at_compute_tier() -> Compute<Self>;

    /// Fused sinh and cosh from one shared pair `exp(x)`, `exp(-x)`.
    ///
    /// A ceiling exp (saturated downscale or overflow sentinel) would wrap
    /// the cosh add; the sum is pinned at the ceiling instead. cosh already
    /// exceeds the storage tier there, so the caller's downscale reports
    /// `TierOverflow`.
    #[inline]
    fn sinhcosh_at_compute_tier(x: Compute<Self>) -> (Compute<Self>, Compute<Self>) {
        let ep = Self::exp_at_compute_tier(x);
        let en = Self::exp_at_compute_tier(Self::compute_negate(x));
        let sinh_c = Self::compute_halve(Self::compute_subtract(ep, en));
        let cosh_sum = Self::compute_checked_add(ep, en).unwrap_or_else(|_| Self::compute_ceiling());
        let cosh_c = Self::compute_halve(cosh_sum);
        (sinh_c, cosh_c)
    }
}

// n / d truncated to q with remainder r (sign of n); step one unit away
// from zero when 2|r| > |d|, or 2|r| == |d| and the quotient is positive
macro_rules! nearest {
    ($n:expr, $d:expr, $q:expr, $r:expr, $zero:expr, $one:expr) => {{
        let (n, d, q, r) = ($n, $d, $q, $r);
        let positive = (n < $zero) == (d < $zero);
        let r_abs = if r < $zero { $zero - r } else { r };
        let d_abs = if d < $zero { $zero - d } else { d };
        let rest = d_abs - r_abs; // compare |r| with |d| - |r|: no doubling
        let bump = if positive { r_abs >= rest } else { r_abs > rest };
        if !bump { q } else if positive { q + $one } else { q - $one }
    }};
}

/// IEEE 754 bits of `magnitude x 2^-frac_bits`, `magnitude` given as
/// little-endian 64-bit words, in a binary format with `fraction_bits` stored
/// fraction bits and `exponent_bits` exponent bits: rounded to
/// `fraction_bits + 1` significant bits, nearest with ties to even; subnormal
/// or zero below the normal range, infinite above it. Integer operations only.
#[inline(always)]
fn ieee_bits(negative: bool, magnitude: &[u64], frac_bits: i32, fraction_bits: u32, exponent_bits: u32) -> u64 {
    let length = word_bit_length(magnitude);
    if length == 0 {
        return 0;
    }
    let sign = (negative as u64) << (fraction_bits + exponent_bits);
    let bias = (1i32 << (exponent_bits - 1)) - 1;
    let top = length as i32 - 1 - frac_bits; // exponent of the leading bit
    // weight of the lowest bit kept: fraction_bits below the leading bit, but
    // never below the subnormal quantum
    let lowest = (top - fraction_bits as i32).max(1 - bias - fraction_bits as i32);
    let shift = lowest + frac_bits; // bits of the magnitude below the kept ones
    let mut significand = if shift <= 0 {
        // everything is kept: the magnitude has at most fraction_bits + 1 bits
        magnitude[0] << (-shift) as u32
    } else {
        let shift = shift as u32;
        let kept = word_bits_from(magnitude, shift);
        let round = word_bit(magnitude, shift - 1);
        let sticky = word_any_below(magnitude, shift - 1);
        kept + (round && (sticky || kept & 1 == 1)) as u64
    };
    let mut lowest = lowest;
    if significand >> (fraction_bits + 1) != 0 {
        // rounding carried into a new leading bit
        significand >>= 1;
        lowest += 1;
    }
    if significand == 0 {
        return sign;
    }
    if significand >> fraction_bits == 0 {
        return sign | significand; // subnormal: exponent field 0
    }
    let biased = lowest + fraction_bits as i32 + bias;
    let infinite = (1i32 << exponent_bits) - 1;
    if biased >= infinite {
        return sign | ((infinite as u64) << fraction_bits);
    }
    sign | ((biased as u64) << fraction_bits) | (significand & ((1u64 << fraction_bits) - 1))
}

#[inline]
fn word_bit_length(words: &[u64]) -> u32 {
    for i in (0..words.len()).rev() {
        if words[i] != 0 {
            return i as u32 * 64 + 64 - words[i].leading_zeros();
        }
    }
    0
}

#[inline]
fn word_bit(words: &[u64], i: u32) -> bool {
    words.get((i / 64) as usize).map_or(false, |w| (w >> (i % 64)) & 1 == 1)
}

/// Any bit below position `i` set.
#[inline]
fn word_any_below(words: &[u64], i: u32) -> bool {
    let (w, b) = ((i / 64) as usize, i % 64);
    words.iter().take(w.min(words.len())).any(|&x| x != 0)
        || (b > 0 && words.get(w).map_or(false, |&x| x & ((1u64 << b) - 1) != 0))
}

/// The magnitude shifted right by `i`, low 64 bits (callers keep at most 54).
#[inline]
fn word_bits_from(words: &[u64], i: u32) -> u64 {
    let (w, b) = ((i / 64) as usize, i % 64);
    let low = words.get(w).map_or(0, |&x| x >> b);
    let high = if b == 0 { 0 } else { words.get(w + 1).map_or(0, |&x| x << (64 - b)) };
    low | high
}

// ============================================================================
// 32-bit storage, any split: Q(32-F).F on i32, compute tier Q(64-2F).2F on i64
// ============================================================================
//
// The engines run at Q64.64: a compute value is shifted up by 64 - 2F, the
// Q64.64 engine evaluated, and the result rounded back (saturating, so an
// oversized result stays at the i64 ceiling for the storage downscale to
// refuse).

/// The 32-bit width's engine wrappers, generic over the split so that the
/// upshift `64 - 2F` is an immediate, as the old per-profile constant was. A compute value is shifted up to Q64.64, the engine evaluated,
/// and the result rounded back (saturating, so an oversized result stays at
/// the i64 ceiling for the storage downscale to refuse).
mod c32 {
    use super::*;

    #[inline(always)]
    fn up<const F: u32>(val: i64) -> i128 {
        (val as i128) << (64 - 2 * F)
    }

    #[inline(always)]
    fn down<const F: u32>(val: i128) -> i64 {
        const { assert!(F >= 2 && F <= 30) }
        let s = 64 - 2 * F;
        let round_bit = (val & (1i128 << (s - 1))) != 0;
        let mut result = val >> s;
        if round_bit { result += 1; }
        if result > i64::MAX as i128 { i64::MAX }
        else if result < i64::MIN as i128 { i64::MIN }
        else { result as i64 }
    }

    #[inline(always)]
    pub(super) fn exp<const F: u32>(x: i64) -> i64 { down::<F>(exp_q64_64_native(up::<F>(x))) }
    #[inline(always)]
    pub(super) fn ln<const F: u32>(x: i64) -> i64 {
        if x <= 0 {
            return i64::MIN;
        }
        down::<F>(ln_q64_64_native(up::<F>(x)))
    }
    #[inline(always)]
    pub(super) fn sqrt<const F: u32>(x: i64) -> i64 {
        if x < 0 { return i64::MIN; }
        down::<F>(sqrt_q64_64_native(up::<F>(x)))
    }
    #[inline(always)]
    pub(super) fn sin<const F: u32>(x: i64) -> i64 { down::<F>(sincos_q64_64_native(up::<F>(x)).0) }
    #[inline(always)]
    pub(super) fn cos<const F: u32>(x: i64) -> i64 { down::<F>(sincos_q64_64_native(up::<F>(x)).1) }
    #[inline(always)]
    pub(super) fn sincos<const F: u32>(x: i64) -> (i64, i64) {
        let (s, c) = sincos_q64_64_native(up::<F>(x));
        (down::<F>(s), down::<F>(c))
    }
    #[inline(always)]
    pub(super) fn atan<const F: u32>(x: i64) -> i64 { down::<F>(atan_q64_64_native(up::<F>(x))) }
    #[inline(always)]
    pub(super) fn atan2<const F: u32>(y: i64, x: i64) -> i64 { down::<F>(atan2_q64_64_native(up::<F>(y), up::<F>(x))) }
}

impl<const F: u32> sealed::Sealed for Fixed<i32, F> {}
impl<const F: u32> FixedNum for Fixed<i32, F> {
    type Storage = i32;
    #[inline]
    fn from_raw_storage(raw: Self::Storage) -> Self { Self { raw } }
    #[inline]
    fn raw_storage(self) -> Self::Storage { self.raw }
    // Every kernel that depends on the split reads it through this constant,
    // so an unsupported split is a compile error where it is first used.
    const FRAC_BITS: u32 = {
        assert!(F >= 2 && F <= 30, "Fixed<i32, F>: F must be between 2 and 30");
        F
    };
    const STORAGE_TIER: u8 = 1;
    const CONVERGENCE_SHIFT: u32 = 8;
    #[inline]
    fn upscale_to_compute(val: i32) -> i64 {
        // i32 → i64, shift left by FRAC_BITS (storage → compute tier)
        (val as i64) << Self::FRAC_BITS
    }
    #[inline]
    fn downscale_to_storage(val: i64) -> Result<i32, OverflowDetected> {
        // The round bit is added at the compute width BEFORE the range check:
        // checking first let a value that rounds up to 2^(W-1) through, and the
        // bump then overflowed the storage type (a panic in debug builds, a wrap
        // in release; before 0.6.4).
        // i64 → i32, shift right by FRAC_BITS with rounding
        let round_bit = (val & (1i64 << (Self::FRAC_BITS - 1))) != 0;
        let shifted = (val >> Self::FRAC_BITS) + round_bit as i64;
        i32::try_from(shifted).map_err(|_| OverflowDetected::TierOverflow)
    }
    #[inline]
    fn downscale_to_storage_floor(val: i64) -> Result<i32, OverflowDetected> {
        let shifted = val >> Self::FRAC_BITS;
        if shifted > i32::MAX as i64 || shifted < i32::MIN as i64 {
            return Err(OverflowDetected::TierOverflow);
        }
        Ok(shifted as i32)
    }
    #[inline]
    fn downscale_to_storage_ceil(val: i64) -> Result<i32, OverflowDetected> {
        let inexact = (val & ((1i64 << Self::FRAC_BITS) - 1)) != 0;
        let shifted = val >> Self::FRAC_BITS;
        if shifted > i32::MAX as i64 || shifted < i32::MIN as i64 {
            return Err(OverflowDetected::TierOverflow);
        }
        let floor = shifted as i32;
        if inexact {
            floor.checked_add(1).ok_or(OverflowDetected::TierOverflow)
        } else {
            Ok(floor)
        }
    }
    #[inline]
    fn compute_ceiling() -> i64 {
        i64::MAX
    }
    #[inline]
    fn exp_sentinel_reached(v: &i64) -> bool {
        *v == i64::MAX
    }
    fn compute_checked_multiply(a: i64, b: i64) -> Result<i64, OverflowDetected> {
        const OVERFLOW: OverflowDetected = OverflowDetected::TierOverflow;
        // i64 x i64 -> i128 >> COMPUTE_FRAC_BITS -> i64 (with rounding)
        let product = (a as i128) * (b as i128);
        let round_bit = ((product >> (2 * Self::FRAC_BITS - 1)) & 1) as i128;
        i64::try_from((product >> (2 * Self::FRAC_BITS)) + round_bit).map_err(|_| OVERFLOW)
    }
    #[inline]
    fn compute_checked_divide(a: i64, b: i64) -> Result<i64, OverflowDetected> {
        if b == 0i64 { return Err(OverflowDetected::DivisionByZero); }
        let (n, d) = ((a as i128) << (2 * Self::FRAC_BITS), b as i128);
        let q = nearest!(n, d, n / d, n % d, 0i128, 1i128);
        if q > i64::MAX as i128 || q < i64::MIN as i128 { return Err(OverflowDetected::TierOverflow); }
        Ok(q as i64)
    }
    #[inline]
    fn make_compute_int(n: i64) -> i64 {
        n.checked_mul(1i64 << (2 * Self::FRAC_BITS))
            .expect("make_compute_int: integer outside the realtime compute range")
    }
    fn compute_mul_div_int(c: i64, num: i64, den: i64) -> Result<i64, OverflowDetected> {
        if den <= 0 { return Err(OverflowDetected::DivisionByZero); }
        let p = (c as i128) * (num as i128);
        let d = den as i128;
        // floor((2p + d) / 2d): nearest, ties toward +infinity
        let q = (2 * p + d).div_euclid(2 * d);
        i64::try_from(q).map_err(|_| OverflowDetected::TierOverflow)
    }
    fn compute_div_count(c: i64, n: usize) -> Result<i64, OverflowDetected> {
        if n == 0 { return Err(OverflowDetected::DivisionByZero); }
        // one exact quotient, nearest: the rule compute_divide follows
        // (it truncated like compute_divide did before 0.6.4)
        let n = i64::try_from(n).map_err(|_| OverflowDetected::TierOverflow)?;
        Self::compute_mul_div_int(c, 1, n)
    }
    #[inline]
    fn one_storage() -> i32 {
        1i32 << Self::FRAC_BITS
    }
    fn truncated_raw(mantissa: u64, shift: i32, negative: bool) -> Result<i32, OverflowDetected> {
        const WIDTH: u32 = 32;
        let magnitude: u128 = if shift >= 0 {
            // a magnitude of more than 128 bits overflows every width here
            if 64 - mantissa.leading_zeros() as i32 + shift > 128 {
                return Err(OverflowDetected::TierOverflow);
            }
            (mantissa as u128) << shift
        } else if shift > -128 {
            (mantissa as u128) >> (-shift)
        } else {
            0
        };
        let limit = 1u128 << (WIDTH - 1);
        if magnitude > limit || (magnitude == limit && !negative) {
            return Err(OverflowDetected::TierOverflow);
        }
        let value = if negative { (magnitude as i128).wrapping_neg() } else { magnitude as i128 };
        Ok(value as i32)
    }
    #[inline]
    fn try_fixed_multiply(a: i32, b: i32) -> Result<i32, OverflowDetected> {
        let overflow = OverflowDetected::TierOverflow;
        // i32*i32->i64, >>FRAC_BITS with round-bit: nearest, ties toward +inf
        let wide = (a as i64) * (b as i64);
        let round_bit = ((wide >> (Self::FRAC_BITS - 1)) & 1) as i32;
        let q = wide >> Self::FRAC_BITS;
        if q as i32 as i64 != q { return Err(overflow); }
        (q as i32).checked_add(round_bit).ok_or(overflow)
    }
    #[inline]
    fn try_fixed_divide(a: i32, b: i32) -> Result<i32, OverflowDetected> {
        let overflow = OverflowDetected::TierOverflow;
        let num = (a as i64) << Self::FRAC_BITS;
        let den = b as i64;
        if den == 0 { return Err(OverflowDetected::DivisionByZero); }
        let q = num / den;
        let rem2 = (num - q * den).unsigned_abs() << 1;
        let dabs = den.unsigned_abs();
        let positive = (num < 0) == (den < 0);
        let bump = if positive { rem2 >= dabs } else { rem2 > dabs };
        i32::try_from(if bump { q + if positive { 1 } else { -1 } } else { q }).map_err(|_| overflow)
    }
    #[inline]
    fn try_from_int(v: i32) -> Result<i32, OverflowDetected> {
        let raw = (v as i64) << Self::FRAC_BITS;
        i32::try_from(raw).map_err(|_| OverflowDetected::TierOverflow)
    }
    #[inline]
    fn try_to_int(raw: i32) -> Result<i32, OverflowDetected> {
        Ok(raw >> Self::FRAC_BITS)
    }
    #[inline]
    fn float_bits(raw: i32, fraction_bits: u32, exponent_bits: u32) -> u64 {
        let magnitude = (raw as i128).unsigned_abs();
        let words = [magnitude as u64, (magnitude >> 64) as u64];
        ieee_bits(raw < 0, &words, Self::FRAC_BITS as i32, fraction_bits, exponent_bits)
    }

    #[inline]
    fn direct_exp(x: i64) -> i64 { c32::exp::<F>(x) }
    #[inline]
    fn direct_ln(x: i64) -> i64 { c32::ln::<F>(x) }
    #[inline]
    fn direct_sqrt(x: i64) -> i64 { c32::sqrt::<F>(x) }
    #[inline]
    fn direct_sin(x: i64) -> i64 { c32::sin::<F>(x) }
    #[inline]
    fn direct_cos(x: i64) -> i64 { c32::cos::<F>(x) }
    #[inline]
    fn direct_atan(x: i64) -> i64 { c32::atan::<F>(x) }
    #[inline]
    fn direct_atan2(y: i64, x: i64) -> i64 { c32::atan2::<F>(y, x) }
    #[inline]
    fn compute_pi_half() -> i64 {
        // Shift the Q64.64 constant down with nearest rounding.
        let shift = 64 - 2 * Self::FRAC_BITS;
        let q64 = PI_HALF_Q64;
        let round = (q64 >> (shift - 1)) & 1;
        ((q64 >> shift) + round) as i64
    }

    #[inline]
    fn exp_at_compute_tier(x: i64) -> i64 { Self::direct_exp(x) }
    #[inline]
    fn ln_at_compute_tier(x: i64) -> i64 { Self::direct_ln(x) }
    #[inline]
    fn sqrt_at_compute_tier(x: i64) -> i64 { Self::direct_sqrt(x) }
    #[inline]
    fn sin_at_compute_tier(x: i64) -> i64 { Self::direct_sin(x) }
    #[inline]
    fn cos_at_compute_tier(x: i64) -> i64 { Self::direct_cos(x) }
    #[inline]
    fn sincos_at_compute_tier(x: i64) -> (i64, i64) { c32::sincos::<F>(x) }
    #[inline]
    fn atan_at_compute_tier(x: i64) -> i64 { Self::direct_atan(x) }
    #[inline]
    fn atan2_at_compute_tier(y: i64, x: i64) -> i64 { Self::direct_atan2(y, x) }
    #[inline]
    fn pi_half_at_compute_tier() -> i64 {
        // the Q64.64 constant truncated to the compute tier
        (PI_HALF_Q64 >> (64 - 2 * Self::FRAC_BITS)) as i64
    }
}

// ============================================================================
// 64-bit storage: Q32.32 on i64, compute tier Q64.64 on i128
// ============================================================================

impl sealed::Sealed for Fixed<i64, 32> {}
impl FixedNum for Fixed<i64, 32> {
    type Storage = i64;
    #[inline]
    fn from_raw_storage(raw: Self::Storage) -> Self { Self { raw } }
    #[inline]
    fn raw_storage(self) -> Self::Storage { self.raw }
    const FRAC_BITS: u32 = 32;
    const STORAGE_TIER: u8 = 2;
    const CONVERGENCE_SHIFT: u32 = 16;
    #[inline]
    fn upscale_to_compute(val: i64) -> i128 {
        // i64 → i128, shift left 32 (Q32.32 → Q64.64)
        (val as i128) << 32
    }
    #[inline]
    fn downscale_to_storage(val: i128) -> Result<i64, OverflowDetected> {
        // The round bit is added at the compute width BEFORE the range check:
        // checking first let a value that rounds up to 2^(W-1) through, and the
        // bump then overflowed the storage type (a panic in debug builds, a wrap
        // in release; before 0.6.4).
        // i128 → i64, shift right 32 with rounding (Q64.64 → Q32.32)
        let round_bit = (val & (1i128 << 31)) != 0;
        let shifted = (val >> 32) + round_bit as i128;
        i64::try_from(shifted).map_err(|_| OverflowDetected::TierOverflow)
    }
    #[inline]
    fn downscale_to_storage_floor(val: i128) -> Result<i64, OverflowDetected> {
        let shifted = val >> 32;
        if shifted > i64::MAX as i128 || shifted < i64::MIN as i128 {
            return Err(OverflowDetected::TierOverflow);
        }
        Ok(shifted as i64)
    }
    #[inline]
    fn downscale_to_storage_ceil(val: i128) -> Result<i64, OverflowDetected> {
        let inexact = (val & ((1i128 << 32) - 1)) != 0;
        let shifted = val >> 32;
        if shifted > i64::MAX as i128 || shifted < i64::MIN as i128 {
            return Err(OverflowDetected::TierOverflow);
        }
        let floor = shifted as i64;
        if inexact {
            floor.checked_add(1).ok_or(OverflowDetected::TierOverflow)
        } else {
            Ok(floor)
        }
    }
    #[inline]
    fn compute_ceiling() -> i128 {
        i128::MAX
    }
    #[inline]
    fn exp_sentinel_reached(v: &i128) -> bool {
        *v == i128::MAX
    }
    fn compute_checked_multiply(a: i128, b: i128) -> Result<i128, OverflowDetected> {
        const OVERFLOW: OverflowDetected = OverflowDetected::TierOverflow;
        // i128 x i128 -> I256 >> 64 -> i128 (with rounding)
        let product = I256::from_i128(a) * I256::from_i128(b);
        let round_bit = (product & (I256::from_i128(1) << 63)) != I256::zero();
        let mut shifted = product >> 64;
        if round_bit { shifted = shifted + I256::from_i128(1); }
        if !(shifted.fits_in_i128()) { return Err(OVERFLOW); }
        Ok(shifted.as_i128())
    }
    #[inline]
    fn compute_checked_divide(a: i128, b: i128) -> Result<i128, OverflowDetected> {
        if b == 0i128 { return Err(OverflowDetected::DivisionByZero); }
        let (n, d) = (I256::from_i128(a) << 64, I256::from_i128(b));
        let q = n / d;
        let q = nearest!(n, d, q, n - q * d, I256::zero(), I256::from_i128(1));
        if !q.fits_in_i128() { return Err(OverflowDetected::TierOverflow); }
        Ok(q.as_i128())
    }
    #[inline]
    fn make_compute_int(n: i64) -> i128 {
        (n as i128) << 64
    }
    fn compute_mul_div_int(c: i128, num: i64, den: i64) -> Result<i128, OverflowDetected> {
        if den <= 0 { return Err(OverflowDetected::DivisionByZero); }
        Self::compute_divide(Self::compute_multiply(c, Self::make_compute_int(num)), Self::make_compute_int(den))
    }
    fn compute_div_count(c: i128, n: usize) -> Result<i128, OverflowDetected> {
        if n == 0 { return Err(OverflowDetected::DivisionByZero); }
        Self::compute_divide(c, Self::make_compute_int(n as i64))
    }
    #[inline]
    fn one_storage() -> i64 {
        1i64 << 32
    }
    fn truncated_raw(mantissa: u64, shift: i32, negative: bool) -> Result<i64, OverflowDetected> {
        const WIDTH: u32 = 64;
        let magnitude: u128 = if shift >= 0 {
            // a magnitude of more than 128 bits overflows every width here
            if 64 - mantissa.leading_zeros() as i32 + shift > 128 {
                return Err(OverflowDetected::TierOverflow);
            }
            (mantissa as u128) << shift
        } else if shift > -128 {
            (mantissa as u128) >> (-shift)
        } else {
            0
        };
        let limit = 1u128 << (WIDTH - 1);
        if magnitude > limit || (magnitude == limit && !negative) {
            return Err(OverflowDetected::TierOverflow);
        }
        let value = if negative { (magnitude as i128).wrapping_neg() } else { magnitude as i128 };
        Ok(value as i64)
    }
    #[inline]
    fn try_fixed_multiply(a: i64, b: i64) -> Result<i64, OverflowDetected> {
        let overflow = OverflowDetected::TierOverflow;
        // i64*i64->i128, >>32 with round-bit: nearest, ties toward +inf
        let wide = (a as i128) * (b as i128);
        let round_bit = ((wide >> 31) & 1) as i64;
        let q = wide >> 32;
        if q as i64 as i128 != q { return Err(overflow); }
        (q as i64).checked_add(round_bit).ok_or(overflow)
    }
    #[inline]
    fn try_fixed_divide(a: i64, b: i64) -> Result<i64, OverflowDetected> {
        let overflow = OverflowDetected::TierOverflow;
        let num = (a as i128) << 32;
        let den = b as i128;
        if den == 0 { return Err(OverflowDetected::DivisionByZero); }
        let q = num / den;
        let rem2 = (num - q * den).unsigned_abs() << 1;
        let dabs = den.unsigned_abs();
        let positive = (num < 0) == (den < 0);
        let bump = if positive { rem2 >= dabs } else { rem2 > dabs };
        i64::try_from(if bump { q + if positive { 1 } else { -1 } } else { q }).map_err(|_| overflow)
    }
    #[inline]
    fn try_from_int(v: i32) -> Result<i64, OverflowDetected> {
        Ok((v as i64) << 32)
    }
    #[inline]
    fn try_to_int(raw: i64) -> Result<i32, OverflowDetected> {
        i32::try_from(raw >> 32).map_err(|_| OverflowDetected::TierOverflow)
    }
    #[inline]
    fn float_bits(raw: i64, fraction_bits: u32, exponent_bits: u32) -> u64 {
        let magnitude = (raw as i128).unsigned_abs();
        let words = [magnitude as u64, (magnitude >> 64) as u64];
        ieee_bits(raw < 0, &words, 32, fraction_bits, exponent_bits)
    }

    #[inline]
    fn direct_exp(x: i128) -> i128 { exp_q64_64_native(x) }
    #[inline]
    fn direct_ln(x: i128) -> i128 {
        if x <= 0 {
            return i128::MIN;
        }
        ln_q64_64_native(x)
    }
    #[inline]
    fn direct_sqrt(x: i128) -> i128 { sqrt_q64_64_native(x) }
    #[inline]
    fn direct_sin(x: i128) -> i128 { sincos_q64_64_native(x).0 }
    #[inline]
    fn direct_cos(x: i128) -> i128 { sincos_q64_64_native(x).1 }
    #[inline]
    fn direct_atan(x: i128) -> i128 { atan_q64_64_native(x) }
    #[inline]
    fn direct_atan2(y: i128, x: i128) -> i128 { atan2_q64_64_native(y, x) }
    #[inline]
    fn compute_pi_half() -> i128 { PI_HALF_Q64 }

    #[inline]
    fn exp_at_compute_tier(x: i128) -> i128 { Self::direct_exp(x) }
    #[inline]
    fn ln_at_compute_tier(x: i128) -> i128 { Self::direct_ln(x) }
    #[inline]
    fn sqrt_at_compute_tier(x: i128) -> i128 { Self::direct_sqrt(x) }
    #[inline]
    fn sin_at_compute_tier(x: i128) -> i128 { Self::direct_sin(x) }
    #[inline]
    fn cos_at_compute_tier(x: i128) -> i128 { Self::direct_cos(x) }
    #[inline]
    fn sincos_at_compute_tier(x: i128) -> (i128, i128) { sincos_q64_64_native(x) }
    #[inline]
    fn atan_at_compute_tier(x: i128) -> i128 { Self::direct_atan(x) }
    #[inline]
    fn atan2_at_compute_tier(y: i128, x: i128) -> i128 { Self::direct_atan2(y, x) }
    #[inline]
    fn pi_half_at_compute_tier() -> i128 { PI_HALF_Q64 }
}

// ============================================================================
// 128-bit storage: Q64.64 on i128, compute tier Q128.128 on I256
// ============================================================================

impl sealed::Sealed for Fixed<i128, 64> {}
impl FixedNum for Fixed<i128, 64> {
    type Storage = i128;
    #[inline]
    fn from_raw_storage(raw: Self::Storage) -> Self { Self { raw } }
    #[inline]
    fn raw_storage(self) -> Self::Storage { self.raw }
    const FRAC_BITS: u32 = 64;
    const STORAGE_TIER: u8 = 3;
    const CONVERGENCE_SHIFT: u32 = 32;
    #[inline]
    fn upscale_to_compute(val: i128) -> I256 {
        // i128 → I256, shift left 64
        I256::from_i128(val) << 64
    }
    #[inline]
    fn downscale_to_storage(val: I256) -> Result<i128, OverflowDetected> {
        // The round bit is added at the compute width BEFORE the range check:
        // checking first let a value that rounds up to 2^(W-1) through, and the
        // bump then overflowed the storage type (a panic in debug builds, a wrap
        // in release; before 0.6.4).
        // I256 → i128, shift right 64 with rounding
        let round_bit = (val & (I256::from_i128(1) << 63)) != I256::zero();
        let mut shifted = val >> 64;
        if round_bit { shifted = shifted + I256::from_i128(1); }
        if !shifted.fits_in_i128() {
            return Err(OverflowDetected::TierOverflow);
        }
        Ok(shifted.as_i128())
    }
    #[inline]
    fn downscale_to_storage_floor(val: I256) -> Result<i128, OverflowDetected> {
        let shifted = val >> 64;
        if !shifted.fits_in_i128() {
            return Err(OverflowDetected::TierOverflow);
        }
        Ok(shifted.as_i128())
    }
    #[inline]
    fn downscale_to_storage_ceil(val: I256) -> Result<i128, OverflowDetected> {
        let mask = (I256::from_i128(1) << 64) - I256::from_i128(1);
        let inexact = (val & mask) != I256::zero();
        let shifted = val >> 64;
        if !shifted.fits_in_i128() {
            return Err(OverflowDetected::TierOverflow);
        }
        let floor = shifted.as_i128();
        if inexact {
            floor.checked_add(1).ok_or(OverflowDetected::TierOverflow)
        } else {
            Ok(floor)
        }
    }
    #[inline]
    fn compute_ceiling() -> I256 {
        I256::max_value()
    }
    #[inline]
    fn exp_sentinel_reached(v: &I256) -> bool {
        *v >= I256::from_i128(i128::MAX) << 128usize
    }
    fn compute_checked_multiply(a: I256, b: I256) -> Result<I256, OverflowDetected> {
        const OVERFLOW: OverflowDetected = OverflowDetected::TierOverflow;
        // I256 x I256 -> I512 >> 128 -> I256 (with rounding)
        let product = I512::from_i256(a) * I512::from_i256(b);
        let round_bit = (product & (I512::from_i128(1) << 127)) != I512::zero();
        let mut shifted = product >> 128;
        if round_bit { shifted = shifted + I512::from_i128(1); }
        if !(shifted.fits_in_i256()) { return Err(OVERFLOW); }
        Ok(shifted.as_i256())
    }
    #[inline]
    fn compute_checked_divide(a: I256, b: I256) -> Result<I256, OverflowDetected> {
        if b == I256::zero() { return Err(OverflowDetected::DivisionByZero); }
        let (n, d) = (I512::from_i256(a) << 128, I512::from_i256(b));
        let q = n / d;
        let q = nearest!(n, d, q, n - q * d, I512::zero(), I512::from_i128(1));
        if !q.fits_in_i256() { return Err(OverflowDetected::TierOverflow); }
        Ok(q.as_i256())
    }
    #[inline]
    fn make_compute_int(n: i64) -> I256 {
        I256::from_i128(n as i128) << 128
    }
    fn compute_mul_div_int(c: I256, num: i64, den: i64) -> Result<I256, OverflowDetected> {
        if den <= 0 { return Err(OverflowDetected::DivisionByZero); }
        Self::compute_divide(Self::compute_multiply(c, Self::make_compute_int(num)), Self::make_compute_int(den))
    }
    fn compute_div_count(c: I256, n: usize) -> Result<I256, OverflowDetected> {
        if n == 0 { return Err(OverflowDetected::DivisionByZero); }
        Self::compute_divide(c, Self::make_compute_int(n as i64))
    }
    #[inline]
    fn one_storage() -> i128 {
        1i128 << 64
    }
    fn truncated_raw(mantissa: u64, shift: i32, negative: bool) -> Result<i128, OverflowDetected> {
        const WIDTH: u32 = 128;
        let magnitude: u128 = if shift >= 0 {
            // a magnitude of more than 128 bits overflows every width here
            if 64 - mantissa.leading_zeros() as i32 + shift > 128 {
                return Err(OverflowDetected::TierOverflow);
            }
            (mantissa as u128) << shift
        } else if shift > -128 {
            (mantissa as u128) >> (-shift)
        } else {
            0
        };
        let limit = 1u128 << (WIDTH - 1);
        if magnitude > limit || (magnitude == limit && !negative) {
            return Err(OverflowDetected::TierOverflow);
        }
        let value = if negative { (magnitude as i128).wrapping_neg() } else { magnitude as i128 };
        Ok(value as i128)
    }
    #[inline]
    fn try_fixed_multiply(a: i128, b: i128) -> Result<i128, OverflowDetected> {
        let overflow = OverflowDetected::TierOverflow;
        // i128*i128->I256, >>64 plus the round bit (bit 63 of the product):
        // the rounding of multiply_binary_i128, range-checked
        let product = crate::fixed_point::i256::mul_i128_to_i256(a, b);
        let mut scaled = product >> 64u32;
        if product.words[0] >= 1u64 << 63 { scaled = scaled + I256::from_i128(1); }
        if !scaled.fits_in_i128() { return Err(overflow); }
        Ok(scaled.as_i128())
    }
    #[inline]
    fn try_fixed_divide(a: i128, b: i128) -> Result<i128, OverflowDetected> {
        let overflow = OverflowDetected::TierOverflow;
        let num = I256::from_i128(a) << 64usize;
        let den = I256::from_i128(b);
        if den.is_zero() { return Err(OverflowDetected::DivisionByZero); }
        let q = num / den;
        let rem = num - q * den;
        let rem_abs = if rem.is_negative() { -rem } else { rem };
        let den_abs = if den.is_negative() { -den } else { den };
        let positive = num.is_negative() == den.is_negative();
        let rem2 = rem_abs + rem_abs;
        let bump = if positive { rem2 >= den_abs } else { rem2 > den_abs };
        let one = I256::from_i128(1);
        let q = if bump { if positive { q + one } else { q - one } } else { q };
        if !q.fits_in_i128() { return Err(overflow); }
        Ok(q.as_i128())
    }
    #[inline]
    fn try_from_int(v: i32) -> Result<i128, OverflowDetected> {
        Ok((v as i128) << 64)
    }
    #[inline]
    fn try_to_int(raw: i128) -> Result<i32, OverflowDetected> {
        i32::try_from(raw >> 64).map_err(|_| OverflowDetected::TierOverflow)
    }
    #[inline]
    fn float_bits(raw: i128, fraction_bits: u32, exponent_bits: u32) -> u64 {
        let magnitude = (raw as i128).unsigned_abs();
        let words = [magnitude as u64, (magnitude >> 64) as u64];
        ieee_bits(raw < 0, &words, 64, fraction_bits, exponent_bits)
    }

    #[inline]
    fn direct_exp(x: I256) -> I256 { exp_q128_128_native(x) }
    #[inline]
    fn direct_ln(x: I256) -> I256 {
        if x <= I256::zero() {
            return I256::min_value();
        }
        ln_q128_128_native(x)
    }
    #[inline]
    fn direct_sqrt(x: I256) -> I256 { sqrt_q128_128_native(x) }
    #[inline]
    fn direct_sin(x: I256) -> I256 { sincos_q128_128_native(x).0 }
    #[inline]
    fn direct_cos(x: I256) -> I256 { sincos_q128_128_native(x).1 }
    // the full-precision engine here, the compute-tier one on the canonical
    // path below: both as they always were
    #[inline]
    fn direct_atan(x: I256) -> I256 { atan_q128_128_native(x) }
    #[inline]
    fn direct_atan2(y: I256, x: I256) -> I256 { atan2_q128_128_compute(y, x) }
    #[inline]
    fn compute_pi_half() -> I256 {
        // the Q64.64 constant at the compute tier (its low 64 bits are zero)
        Self::upscale_to_compute(PI_HALF_Q64)
    }

    #[inline]
    fn exp_at_compute_tier(x: I256) -> I256 { Self::direct_exp(x) }
    #[inline]
    fn ln_at_compute_tier(x: I256) -> I256 { Self::direct_ln(x) }
    #[inline]
    fn sqrt_at_compute_tier(x: I256) -> I256 { Self::direct_sqrt(x) }
    #[inline]
    fn sin_at_compute_tier(x: I256) -> I256 { Self::direct_sin(x) }
    #[inline]
    fn cos_at_compute_tier(x: I256) -> I256 { Self::direct_cos(x) }
    #[inline]
    fn sincos_at_compute_tier(x: I256) -> (I256, I256) { sincos_q128_128_native(x) }
    #[inline]
    fn atan_at_compute_tier(x: I256) -> I256 { atan_q128_128_compute(x) }
    #[inline]
    fn atan2_at_compute_tier(y: I256, x: I256) -> I256 { atan2_q128_128_compute(y, x) }
    #[inline]
    fn pi_half_at_compute_tier() -> I256 { I256::from_words(PI_HALF_Q128) }
}

// ============================================================================
// 256-bit storage: Q128.128 on I256, compute tier Q256.256 on I512
// ============================================================================

impl sealed::Sealed for Fixed<I256, 128> {}
impl FixedNum for Fixed<I256, 128> {
    type Storage = I256;
    #[inline]
    fn from_raw_storage(raw: Self::Storage) -> Self { Self { raw } }
    #[inline]
    fn raw_storage(self) -> Self::Storage { self.raw }
    const FRAC_BITS: u32 = 128;
    const STORAGE_TIER: u8 = 4;
    const CONVERGENCE_SHIFT: u32 = 64;
    #[inline]
    fn upscale_to_compute(val: I256) -> I512 {
        // I256 → I512, shift left 128
        I512::from_i256(val) << 128
    }
    #[inline]
    fn downscale_to_storage(val: I512) -> Result<I256, OverflowDetected> {
        // The round bit is added at the compute width BEFORE the range check:
        // checking first let a value that rounds up to 2^(W-1) through, and the
        // bump then overflowed the storage type (a panic in debug builds, a wrap
        // in release; before 0.6.4).
        // I512 → I256, shift right 128 with rounding
        let round_bit = (val & (I512::from_i128(1) << 127)) != I512::zero();
        let mut shifted = val >> 128;
        if round_bit { shifted = shifted + I512::from_i128(1); }
        if !shifted.fits_in_i256() {
            return Err(OverflowDetected::TierOverflow);
        }
        Ok(shifted.as_i256())
    }
    #[inline]
    fn downscale_to_storage_floor(val: I512) -> Result<I256, OverflowDetected> {
        let shifted = val >> 128;
        if !shifted.fits_in_i256() {
            return Err(OverflowDetected::TierOverflow);
        }
        Ok(shifted.as_i256())
    }
    #[inline]
    fn downscale_to_storage_ceil(val: I512) -> Result<I256, OverflowDetected> {
        let mask = (I512::from_i128(1) << 128) - I512::from_i128(1);
        let inexact = (val & mask) != I512::zero();
        let shifted = val >> 128;
        if !shifted.fits_in_i256() {
            return Err(OverflowDetected::TierOverflow);
        }
        let floor = shifted.as_i256();
        if inexact {
            floor.checked_add(I256::from_i128(1)).ok_or(OverflowDetected::TierOverflow)
        } else {
            Ok(floor)
        }
    }
    #[inline]
    fn compute_ceiling() -> I512 {
        I512::max_value()
    }
    #[inline]
    fn exp_sentinel_reached(v: &I512) -> bool {
        *v >= I512::from_i256(I256::from_i128(i128::MAX)) << 256usize
    }
    fn compute_checked_multiply(a: I512, b: I512) -> Result<I512, OverflowDetected> {
        const OVERFLOW: OverflowDetected = OverflowDetected::TierOverflow;
        // I512 x I512 -> I1024 >> 256 -> I512 (with rounding)
        let product = I1024::from_i512(a) * I1024::from_i512(b);
        let round_bit = (product & (I1024::from_i128(1) << 255)) != I1024::zero();
        let mut shifted = product >> 256;
        if round_bit { shifted = shifted + I1024::from_i128(1); }
        if !(shifted.fits_in_i512()) { return Err(OVERFLOW); }
        Ok(shifted.as_i512())
    }
    #[inline]
    fn compute_checked_divide(a: I512, b: I512) -> Result<I512, OverflowDetected> {
        if b == I512::zero() { return Err(OverflowDetected::DivisionByZero); }
        let (n, d) = (I1024::from_i512(a) << 256, I1024::from_i512(b));
        let q = n / d;
        let q = nearest!(n, d, q, n - q * d, I1024::zero(), I1024::from_i128(1));
        if !q.fits_in_i512() { return Err(OverflowDetected::TierOverflow); }
        Ok(q.as_i512())
    }
    #[inline]
    fn make_compute_int(n: i64) -> I512 {
        I512::from_i128(n as i128) << 256
    }
    fn compute_mul_div_int(c: I512, num: i64, den: i64) -> Result<I512, OverflowDetected> {
        if den <= 0 { return Err(OverflowDetected::DivisionByZero); }
        Self::compute_divide(Self::compute_multiply(c, Self::make_compute_int(num)), Self::make_compute_int(den))
    }
    fn compute_div_count(c: I512, n: usize) -> Result<I512, OverflowDetected> {
        if n == 0 { return Err(OverflowDetected::DivisionByZero); }
        Self::compute_divide(c, Self::make_compute_int(n as i64))
    }
    #[inline]
    fn one_storage() -> I256 {
        crate::fixed_point::i256::I256::from_i128(1) << 128
    }
    fn truncated_raw(mantissa: u64, shift: i32, negative: bool) -> Result<I256, OverflowDetected> {
        let (width, one, zero) = (256i32, I256::from_i128(1), I256::zero());
        let magnitude = if shift < 0 {
            if shift <= -64 { zero } else { I256::from_i128((mantissa >> (-shift)) as i128) }
        } else {
            let length = 64 - mantissa.leading_zeros() as i32 + shift;
            if length < width {
                I256::from_i128((mantissa) as i128) << (shift as usize)
            } else if length == width && negative && mantissa.is_power_of_two() {
                // exactly 2^(W-1): the storage minimum
                return Ok(one << ((width - 1) as usize));
            } else {
                return Err(OverflowDetected::TierOverflow);
            }
        };
        Ok(if negative { -magnitude } else { magnitude })
    }
    #[inline]
    fn try_fixed_multiply(a: I256, b: I256) -> Result<I256, OverflowDetected> {
        let overflow = OverflowDetected::TierOverflow;
        // I256*I256->I512 on magnitudes (mul_to_i512 is unsigned), read by
        // words: the magnitude before rounding is words 2..6, the remainder
        // words 0..2. A positive result rounds the magnitude up on
        // remainder >= half (tie up), a negative one only on remainder > half
        // (tie toward +inf).
        let a_neg = a.is_negative();
        let b_neg = b.is_negative();
        let result_neg = a_neg != b_neg;
        let abs_a = if a_neg { -a } else { a };
        let abs_b = if b_neg { -b } else { b };
        let w = abs_a.mul_to_i512(abs_b).words;
        let top = 1u64 << 63;
        let bump = if result_neg { w[1] > top || (w[1] == top && w[0] != 0) } else { w[1] >= top };
        if w[6] != 0 || w[7] != 0 || w[5] > top {
            return Err(overflow);
        }
        // round: add the bump through the carry chain (the top word is at
        // most 2^63, so it cannot carry out)
        let mut m = [w[2], w[3], w[4], w[5]];
        let mut carry = bump as u64;
        for x in m.iter_mut() {
            let (v, c) = x.overflowing_add(carry);
            *x = v;
            carry = c as u64;
        }
        if m[3] >= top {
            // 2^255 or more: only the negative minimum, magnitude exactly 2^255
            return if result_neg && m == [0, 0, 0, top] { Ok(I256::min_value()) } else { Err(overflow) };
        }
        let mag = I256 { words: m };
        Ok(if result_neg { -mag } else { mag })
    }
    #[inline]
    fn try_fixed_divide(a: I256, b: I256) -> Result<I256, OverflowDetected> {
        let overflow = OverflowDetected::TierOverflow;
        let num = I512::from_i256(a) << 128usize;
        let den = I512::from_i256(b);
        if den.is_zero() { return Err(OverflowDetected::DivisionByZero); }
        let q = num / den;
        let rem = num - q * den;
        let rem_abs = if rem.is_negative() { -rem } else { rem };
        let den_abs = if den.is_negative() { -den } else { den };
        let positive = num.is_negative() == den.is_negative();
        let rem2 = rem_abs + rem_abs;
        let bump = if positive { rem2 >= den_abs } else { rem2 > den_abs };
        let one = I512::from_i128(1);
        let q = if bump { if positive { q + one } else { q - one } } else { q };
        if !q.fits_in_i256() { return Err(overflow); }
        Ok(q.as_i256())
    }
    #[inline]
    fn try_from_int(v: i32) -> Result<I256, OverflowDetected> {
        Ok(I256::from_i128(v as i128) << 128usize)
    }
    #[inline]
    fn try_to_int(raw: I256) -> Result<i32, OverflowDetected> {
        i32::try_from((raw >> 128u32).as_i128()).map_err(|_| OverflowDetected::TierOverflow)
    }
    #[inline]
    fn float_bits(raw: I256, fraction_bits: u32, exponent_bits: u32) -> u64 {
        let negative = raw.is_negative();
        // two's complement negation wraps at the minimum, whose words then
        // read as the unsigned magnitude 2^(W-1)
        let magnitude = if negative { -raw } else { raw };
        ieee_bits(negative, &magnitude.words, 128, fraction_bits, exponent_bits)
    }

    #[inline]
    fn direct_exp(x: I512) -> I512 { exp_q256_256_native(x) }
    #[inline]
    fn direct_ln(x: I512) -> I512 {
        if x <= I512::zero() {
            return I512::min_value();
        }
        ln_q256_256_native(x)
    }
    #[inline]
    fn direct_sqrt(x: I512) -> I512 { sqrt_q256_256_native(x) }
    #[inline]
    fn direct_sin(x: I512) -> I512 { sincos_q256_256_native(x).0 }
    #[inline]
    fn direct_cos(x: I512) -> I512 { sincos_q256_256_native(x).1 }
    #[inline]
    fn direct_atan(x: I512) -> I512 { atan_q256_256_compute(x) }
    #[inline]
    fn direct_atan2(y: I512, x: I512) -> I512 { atan2_q256_256_compute(y, x) }
    #[inline]
    fn compute_pi_half() -> I512 {
        // the Q128.128 constant at the compute tier (its low 128 bits are zero)
        Self::upscale_to_compute(I256::from_words(PI_HALF_Q128))
    }

    #[inline]
    fn exp_at_compute_tier(x: I512) -> I512 { Self::direct_exp(x) }
    #[inline]
    fn ln_at_compute_tier(x: I512) -> I512 { Self::direct_ln(x) }
    #[inline]
    fn sqrt_at_compute_tier(x: I512) -> I512 { Self::direct_sqrt(x) }
    #[inline]
    fn sin_at_compute_tier(x: I512) -> I512 { Self::direct_sin(x) }
    #[inline]
    fn cos_at_compute_tier(x: I512) -> I512 { Self::direct_cos(x) }
    #[inline]
    fn sincos_at_compute_tier(x: I512) -> (I512, I512) { sincos_q256_256_native(x) }
    #[inline]
    fn atan_at_compute_tier(x: I512) -> I512 { Self::direct_atan(x) }
    #[inline]
    fn atan2_at_compute_tier(y: I512, x: I512) -> I512 { Self::direct_atan2(y, x) }
    #[inline]
    fn pi_half_at_compute_tier() -> I512 { I512::from_words(PI_HALF_Q256) }
}

// ============================================================================
// 512-bit storage: Q256.256 on I512, compute tier Q512.512 on I1024
// ============================================================================

impl sealed::Sealed for Fixed<I512, 256> {}
impl FixedNum for Fixed<I512, 256> {
    type Storage = I512;
    #[inline]
    fn from_raw_storage(raw: Self::Storage) -> Self { Self { raw } }
    #[inline]
    fn raw_storage(self) -> Self::Storage { self.raw }
    const FRAC_BITS: u32 = 256;
    const STORAGE_TIER: u8 = 5;
    const CONVERGENCE_SHIFT: u32 = 128;
    #[inline]
    fn upscale_to_compute(val: I512) -> I1024 {
        // I512 → I1024, shift left 256
        I1024::from_i512(val) << 256
    }
    #[inline]
    fn downscale_to_storage(val: I1024) -> Result<I512, OverflowDetected> {
        // The round bit is added at the compute width BEFORE the range check:
        // checking first let a value that rounds up to 2^(W-1) through, and the
        // bump then overflowed the storage type (a panic in debug builds, a wrap
        // in release; before 0.6.4).
        // I1024 → I512, shift right 256 with rounding
        let round_bit = (val & (I1024::from_i128(1) << 255)) != I1024::zero();
        let mut shifted = val >> 256;
        if round_bit { shifted = shifted + I1024::from_i128(1); }
        if !shifted.fits_in_i512() {
            return Err(OverflowDetected::TierOverflow);
        }
        Ok(shifted.as_i512())
    }
    #[inline]
    fn downscale_to_storage_floor(val: I1024) -> Result<I512, OverflowDetected> {
        let shifted = val >> 256;
        if !shifted.fits_in_i512() {
            return Err(OverflowDetected::TierOverflow);
        }
        Ok(shifted.as_i512())
    }
    #[inline]
    fn downscale_to_storage_ceil(val: I1024) -> Result<I512, OverflowDetected> {
        let mask = (I1024::from_i128(1) << 256) - I1024::from_i128(1);
        let inexact = (val & mask) != I1024::zero();
        let shifted = val >> 256;
        if !shifted.fits_in_i512() {
            return Err(OverflowDetected::TierOverflow);
        }
        let floor = shifted.as_i512();
        if inexact {
            floor.checked_add(I512::from_i128(1)).ok_or(OverflowDetected::TierOverflow)
        } else {
            Ok(floor)
        }
    }
    #[inline]
    fn compute_ceiling() -> I1024 {
        I1024::max_value()
    }
    #[inline]
    fn exp_sentinel_reached(v: &I1024) -> bool {
        *v == I1024::max_value()
    }
    fn compute_checked_multiply(a: I1024, b: I1024) -> Result<I1024, OverflowDetected> {
        const OVERFLOW: OverflowDetected = OverflowDetected::TierOverflow;
        // I1024 x I1024 -> I2048 >> 512 on magnitudes, as multiply_i1024_q512_512
        // (mul_to_i2048 is unsigned), with the narrowing checked
        let a_negative = a < I1024::zero();
        let b_negative = b < I1024::zero();
        let abs_a = if a_negative { I1024::zero() - a } else { a };
        let abs_b = if b_negative { I1024::zero() - b } else { b };
        let product = abs_a.mul_to_i2048(abs_b);
        let rounded = (product + (I2048::from_i128(1) << 511)) >> 512;
        if !(rounded.fits_in_i1024()) { return Err(OVERFLOW); }
        let magnitude = rounded.as_i1024();
        Ok(if a_negative != b_negative { I1024::zero() - magnitude } else { magnitude })
    }
    #[inline]
    fn compute_checked_divide(a: I1024, b: I1024) -> Result<I1024, OverflowDetected> {
        if b == I1024::zero() { return Err(OverflowDetected::DivisionByZero); }
        use crate::fixed_point::domains::binary_fixed::i2048::i2048_divmod;
        let (n, d) = (I2048::from_i1024(a) << 512, I2048::from_i1024(b));
        let (q, r) = i2048_divmod(n, d);
        let q = nearest!(n, d, q, r, I2048::zero(), I2048::from_i128(1));
        if !q.fits_in_i1024() { return Err(OverflowDetected::TierOverflow); }
        Ok(q.as_i1024())
    }
    #[inline]
    fn make_compute_int(n: i64) -> I1024 {
        I1024::from_i128(n as i128) << 512
    }
    fn compute_mul_div_int(c: I1024, num: i64, den: i64) -> Result<I1024, OverflowDetected> {
        if den <= 0 { return Err(OverflowDetected::DivisionByZero); }
        Self::compute_divide(Self::compute_multiply(c, Self::make_compute_int(num)), Self::make_compute_int(den))
    }
    fn compute_div_count(c: I1024, n: usize) -> Result<I1024, OverflowDetected> {
        if n == 0 { return Err(OverflowDetected::DivisionByZero); }
        Self::compute_divide(c, Self::make_compute_int(n as i64))
    }
    #[inline]
    fn one_storage() -> I512 {
        crate::fixed_point::i512::I512::from_i128(1) << 256
    }
    fn truncated_raw(mantissa: u64, shift: i32, negative: bool) -> Result<I512, OverflowDetected> {
        let (width, one, zero) = (512i32, I512::from_i128(1), I512::zero());
        let magnitude = if shift < 0 {
            if shift <= -64 { zero } else { I512::from_i128((mantissa >> (-shift)) as i128) }
        } else {
            let length = 64 - mantissa.leading_zeros() as i32 + shift;
            if length < width {
                I512::from_i128((mantissa) as i128) << (shift as usize)
            } else if length == width && negative && mantissa.is_power_of_two() {
                // exactly 2^(W-1): the storage minimum
                return Ok(one << ((width - 1) as usize));
            } else {
                return Err(OverflowDetected::TierOverflow);
            }
        };
        Ok(if negative { -magnitude } else { magnitude })
    }
    #[inline]
    fn try_fixed_multiply(a: I512, b: I512) -> Result<I512, OverflowDetected> {
        let overflow = OverflowDetected::TierOverflow;
        // I512*I512->I1024 on magnitudes, read by words (see q128_128): the
        // magnitude before rounding is words 4..12, the remainder words 0..4
        let a_neg = a.is_negative();
        let b_neg = b.is_negative();
        let result_neg = a_neg != b_neg;
        let abs_a = if a_neg { -a } else { a };
        let abs_b = if b_neg { -b } else { b };
        let w = abs_a.mul_to_i1024(abs_b).words;
        let top = 1u64 << 63;
        let bump = if result_neg {
            w[3] > top || (w[3] == top && (w[0] | w[1] | w[2]) != 0)
        } else {
            w[3] >= top
        };
        if w[12..].iter().any(|&x| x != 0) || w[11] > top {
            return Err(overflow);
        }
        let mut m = [0u64; 8];
        m.copy_from_slice(&w[4..12]);
        let mut carry = bump as u64;
        for x in m.iter_mut() {
            let (v, c) = x.overflowing_add(carry);
            *x = v;
            carry = c as u64;
        }
        if m[7] >= top {
            // 2^511 or more: only the negative minimum, magnitude exactly 2^511
            let exact_min = m[7] == top && m[..7].iter().all(|&x| x == 0);
            return if result_neg && exact_min { Ok(I512::min_value()) } else { Err(overflow) };
        }
        let mag = I512 { words: m };
        Ok(if result_neg { -mag } else { mag })
    }
    #[inline]
    fn try_fixed_divide(a: I512, b: I512) -> Result<I512, OverflowDetected> {
        let overflow = OverflowDetected::TierOverflow;
        let num = I1024::from_i512(a) << 256usize;
        let den = I1024::from_i512(b);
        if den.is_zero() { return Err(OverflowDetected::DivisionByZero); }
        let q = num / den;
        let rem = num - q * den;
        let rem_abs = if rem < I1024::zero() { -rem } else { rem };
        let den_abs = if den < I1024::zero() { -den } else { den };
        let positive = (num < I1024::zero()) == (den < I1024::zero());
        let rem2 = rem_abs + rem_abs;
        let bump = if positive { rem2 >= den_abs } else { rem2 > den_abs };
        let one = I1024::from_i128(1);
        let q = if bump { if positive { q + one } else { q - one } } else { q };
        if !q.fits_in_i512() { return Err(overflow); }
        Ok(q.as_i512())
    }
    #[inline]
    fn try_from_int(v: i32) -> Result<I512, OverflowDetected> {
        Ok(I512::from_i128(v as i128) << 256usize)
    }
    #[inline]
    fn try_to_int(raw: I512) -> Result<i32, OverflowDetected> {
        let floor = raw >> 256usize;
        if !floor.fits_in_i128() {
            return Err(OverflowDetected::TierOverflow);
        }
        i32::try_from(floor.as_i128()).map_err(|_| OverflowDetected::TierOverflow)
    }
    #[inline]
    fn float_bits(raw: I512, fraction_bits: u32, exponent_bits: u32) -> u64 {
        let negative = raw.is_negative();
        // two's complement negation wraps at the minimum, whose words then
        // read as the unsigned magnitude 2^(W-1)
        let magnitude = if negative { -raw } else { raw };
        ieee_bits(negative, &magnitude.words, 256, fraction_bits, exponent_bits)
    }

    // the form this width always used: negate, then a checked add
    #[inline]
    fn compute_subtract(a: I1024, b: I1024) -> I1024 {
        a.checked_add(Self::compute_negate(b)).expect("compute tier: subtraction overflow")
    }

    #[inline]
    fn direct_exp(x: I1024) -> I1024 { exp_q512_512_native(x) }
    #[inline]
    fn direct_ln(x: I1024) -> I1024 {
        if x <= I1024::zero() {
            return I1024::min_value();
        }
        ln_q512_512_native(x)
    }
    #[inline]
    fn direct_sqrt(x: I1024) -> I1024 { sqrt_q512_512_native(x) }
    #[inline]
    fn direct_sin(x: I1024) -> I1024 { sincos_q512_512_native(x).0 }
    #[inline]
    fn direct_cos(x: I1024) -> I1024 { sincos_q512_512_native(x).1 }
    #[inline]
    fn direct_atan(x: I1024) -> I1024 { atan_q512_512_native(x) }
    #[inline]
    fn direct_atan2(y: I1024, x: I1024) -> I1024 { atan2_q512_512_native(y, x) }
    #[inline]
    fn compute_pi_half() -> I1024 { I1024::from_words(PI_HALF_Q512) }

    #[inline]
    fn exp_at_compute_tier(x: I1024) -> I1024 { Self::direct_exp(x) }
    #[inline]
    fn ln_at_compute_tier(x: I1024) -> I1024 { Self::direct_ln(x) }
    #[inline]
    fn sqrt_at_compute_tier(x: I1024) -> I1024 { Self::direct_sqrt(x) }
    #[inline]
    fn sin_at_compute_tier(x: I1024) -> I1024 { Self::direct_sin(x) }
    #[inline]
    fn cos_at_compute_tier(x: I1024) -> I1024 { Self::direct_cos(x) }
    #[inline]
    fn sincos_at_compute_tier(x: I1024) -> (I1024, I1024) { sincos_q512_512_native(x) }
    #[inline]
    fn atan_at_compute_tier(x: I1024) -> I1024 { Self::direct_atan(x) }
    #[inline]
    fn atan2_at_compute_tier(y: I1024, x: I1024) -> I1024 { Self::direct_atan2(y, x) }
    #[inline]
    fn pi_half_at_compute_tier() -> I1024 { I1024::from_words(PI_HALF_Q512) }
}
