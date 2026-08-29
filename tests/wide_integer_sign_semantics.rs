//! Sign semantics of the wide integer types: the permanent gate.
//!
//! Two silent-wrong-value bugs found on 2026-08-29 while building certified
//! decimal intervals, both in code that had never been asked the question:
//!
//! - `D256::Sub` and `D512::Sub` never propagated a borrow. The borrow test
//!   compared a `u128` against `u128::MAX`, which is never true, so
//!   `0 - 1` produced `2^64 - 1` (positive) and `2^128 - 1` lost a word.
//!   Used by the UGOD decimal tier 5 and 6 subtraction arms
//!   (`decimal_addition.rs`); whether canonical literals can reach those
//!   arms today is a separate question (they currently parse such magnitudes
//!   into the binary domain), so this gate asks the types directly.
//! - `Ord` on `I1024`, `I2048`, `D256` and `D512` reversed the word
//!   comparison for two negative operands. Two's-complement negatives already
//!   order correctly as unsigned words (`-1 = 0xFF..F > 0xFF..E = -2`), so
//!   the reversal made `-2 > -1`. `I256` and `I512` had been corrected
//!   earlier; the other four had not. On the scientific profile the compute
//!   tier is `I1024`, so any `<`, `min` or `max` between two negative
//!   compute-tier values there was wrong.
//!
//! Profile-independent: these types exist on every profile.

use g_math::fixed_point::domains::decimal_fixed::{D256, D512};
use g_math::fixed_point::{I1024, I2048, I256, I512};
use std::cmp::Ordering;

// ----------------------------------------------------------------------------
// Ordering of negatives, same sign, including values beyond i128
// ----------------------------------------------------------------------------

macro_rules! ord_gate {
    ($name:ident, $t:ty, $from:expr, $neg:expr, $shl:expr) => {
        #[test]
        fn $name() {
            let from = $from;
            let neg = $neg;
            let shl = $shl;
            // small negatives
            assert_eq!(from(-2).cmp(&from(-1)), Ordering::Less, "-2 < -1");
            assert_eq!(from(-1).cmp(&from(-2)), Ordering::Greater, "-1 > -2");
            assert_eq!(from(-1).cmp(&from(-1)), Ordering::Equal);
            assert_eq!(from(-1).cmp(&from(0)), Ordering::Less);
            assert_eq!(from(i128::MIN).cmp(&from(i128::MIN + 1)), Ordering::Less);
            // negatives beyond i128: -(2^130) < -(2^129) < -(2^128 + 1)
            let big = |e: u32| neg(shl(from(1), e));
            assert_eq!(big(130).cmp(&big(129)), Ordering::Less, "-(2^130) < -(2^129)");
            assert_eq!(big(129).cmp(&big(130)), Ordering::Greater);
            assert_eq!(big(128).cmp(&from(-1)), Ordering::Less, "-(2^128) < -1");
            // a sort of mixed signs comes out ascending
            let mut v = vec![from(3), big(129), from(-5), from(0), big(130), from(-1), shl(from(1), 129)];
            v.sort();
            let expect = vec![big(130), big(129), from(-5), from(-1), from(0), from(3), shl(from(1), 129)];
            assert_eq!(v, expect, "mixed-sign sort order");
            // min/max, the operation that exposed the bug
            let a = big(129);
            let b = big(130);
            assert_eq!(std::cmp::min(a, b), b);
            assert_eq!(std::cmp::max(a, b), a);
        }
    };
}

ord_gate!(i256_orders_negatives, I256, |x: i128| I256::from_i128(x), |v: I256| -v, |v: I256, e: u32| v << (e as usize));
ord_gate!(i512_orders_negatives, I512, |x: i128| I512::from_i128(x), |v: I512| -v, |v: I512, e: u32| v << (e as usize));
ord_gate!(i1024_orders_negatives, I1024, |x: i128| I1024::from_i128(x), |v: I1024| -v, |v: I1024, e: u32| v << (e as usize));
ord_gate!(i2048_orders_negatives, I2048, |x: i128| I2048::from_i128(x), |v: I2048| -v, |v: I2048, e: u32| v << (e as usize));
ord_gate!(
    d256_orders_negatives,
    D256,
    |x: i128| D256::from_i128(x),
    |v: D256| g_math::fixed_point::domains::decimal_fixed::negate_d256(v),
    |v: D256, e: u32| d256_shl(v, e)
);
ord_gate!(
    d512_orders_negatives,
    D512,
    |x: i128| D512::from_i128(x),
    |v: D512| g_math::fixed_point::domains::decimal_fixed::negate_d512(v),
    |v: D512, e: u32| d512_shl(v, e)
);

/// Left shift by whole words plus bits, for the decimal types, which have no
/// shift operator: multiply by 2^e via repeated doubling of the words.
fn d256_shl(v: D256, e: u32) -> D256 {
    let mut w = v.words;
    for _ in 0..e {
        let mut carry = 0u64;
        for word in w.iter_mut() {
            let next = *word >> 63;
            *word = (*word << 1) | carry;
            carry = next;
        }
    }
    D256::from_words(w)
}

fn d512_shl(v: D512, e: u32) -> D512 {
    let mut w = v.words;
    for _ in 0..e {
        let mut carry = 0u64;
        for word in w.iter_mut() {
            let next = *word >> 63;
            *word = (*word << 1) | carry;
            carry = next;
        }
    }
    D512::from_words(w)
}

// ----------------------------------------------------------------------------
// Subtraction with borrow across words
// ----------------------------------------------------------------------------

#[test]
fn d256_sub_propagates_borrow() {
    // 0 - 1 = -1 (all ones), not 2^64 - 1
    assert_eq!(D256::from_i128(0) - D256::from_i128(1), D256::from_i128(-1));
    // 2^64 - 1: borrow from word 1 into word 0
    let two64 = D256::from_words([0, 1, 0, 0]);
    assert_eq!(two64 - D256::from_i128(1), D256::from_words([u64::MAX, 0, 0, 0]));
    // 2^128 - 1: borrow chain through two words
    let two128 = D256::from_words([0, 0, 1, 0]);
    assert_eq!(two128 - D256::from_i128(1), D256::from_words([u64::MAX, u64::MAX, 0, 0]));
    // 2^192 - 2^64: borrow lands on word 1
    let two192 = D256::from_words([0, 0, 0, 1]);
    assert_eq!(two192 - two64, D256::from_words([0, u64::MAX, u64::MAX, 0]));
    // a - b == -(b - a)
    let a = D256::from_words([5, 7, 0, 0]);
    let b = D256::from_words([9, 2, 0, 0]);
    assert_eq!(a - b, g_math::fixed_point::domains::decimal_fixed::negate_d256(b - a));
    // consistency with i128 on in-range operands, both signs
    for (x, y) in [(5i128, 9i128), (-5, 9), (5, -9), (-5, -9), (i128::MIN + 1, 1), (0, i128::MAX)] {
        assert_eq!(D256::from_i128(x) - D256::from_i128(y), D256::from_i128(x - y), "{x} - {y}");
    }
}

#[test]
fn d512_sub_propagates_borrow() {
    assert_eq!(D512::from_i128(0) - D512::from_i128(1), D512::from_i128(-1));
    let two64 = D512::from_words([0, 1, 0, 0, 0, 0, 0, 0]);
    assert_eq!(two64 - D512::from_i128(1), D512::from_words([u64::MAX, 0, 0, 0, 0, 0, 0, 0]));
    let two256 = D512::from_words([0, 0, 0, 0, 1, 0, 0, 0]);
    assert_eq!(
        two256 - D512::from_i128(1),
        D512::from_words([u64::MAX, u64::MAX, u64::MAX, u64::MAX, 0, 0, 0, 0])
    );
    for (x, y) in [(5i128, 9i128), (-5, 9), (5, -9), (-5, -9), (i128::MIN + 1, 1), (0, i128::MAX)] {
        assert_eq!(D512::from_i128(x) - D512::from_i128(y), D512::from_i128(x - y), "{x} - {y}");
    }
}
