//! 0.6.4: the FixedPoint operators are loud at the storage edge. `+ - * /`
//! and unary `-` panic where the result leaves storage (before 0.6.4 they
//! wrapped silently on q16/q32/q64 and truncated the multiply and divide
//! casts); `try_add / try_sub / try_neg / try_mul / try_div` return the error
//! instead. In range the operators and their try twins are bit-identical.
//!
//! Every edge value is built from the build's own storage (doubling to the
//! top bit, the unit from FRAC_BITS), so the gate holds on every profile and
//! every realtime split.

use g_math::fixed_point::frac_config::FRAC_BITS;
use g_math::fixed_point::{FixedMatrix, FixedPoint, FixedVector, OverflowDetected};

fn one() -> FixedPoint { FixedPoint::one() }

/// One storage unit, 2^-FRAC_BITS (exact: halving a power of two).
fn unit() -> FixedPoint {
    // from a literal: at F=30 the value 2 is out of range
    let half = FixedPoint::from_str("0.5");
    let mut u = one();
    for _ in 0..FRAC_BITS { u = u.try_mul(half).unwrap(); }
    u
}

/// The largest power of two in storage, 2^(W-2) raw.
fn top_power() -> FixedPoint {
    let mut p = one();
    while let Ok(q) = p.try_add(p) { p = q; }
    p
}

/// The storage minimum, raw -2^(W-1), and the maximum, raw 2^(W-1) - 1.
fn min() -> FixedPoint { top_power().try_neg().unwrap().try_sub(top_power()).unwrap() }
fn max() -> FixedPoint { top_power().try_sub(unit()).unwrap().try_add(top_power()).unwrap() }

#[test]
fn edges_are_the_storage_edges() {
    // min = -(max + unit): the two's complement ends, built without wrapping
    assert_eq!(min().try_add(max()).unwrap(), unit().try_neg().unwrap());
    assert!(max().try_add(unit()).is_err());
    assert!(min().try_sub(unit()).is_err());
}

#[test]
fn try_twins_report_overflow() {
    assert_eq!(max().try_add(unit()), Err(OverflowDetected::TierOverflow));
    assert_eq!(min().try_sub(unit()), Err(OverflowDetected::TierOverflow));
    assert_eq!(min().try_neg(), Err(OverflowDetected::TierOverflow));
    assert_eq!(max().try_mul(FixedPoint::from_str("1.5")), Err(OverflowDetected::TierOverflow));
    assert_eq!(max().try_mul(max()), Err(OverflowDetected::TierOverflow));
    assert_eq!(min().try_mul(one().try_neg().unwrap()), Err(OverflowDetected::TierOverflow));
    assert_eq!(one().try_div(FixedPoint::ZERO), Err(OverflowDetected::DivisionByZero));
    assert_eq!(min().try_div(one().try_neg().unwrap()), Err(OverflowDetected::TierOverflow));
    // a small divisor pushes the quotient out: max / (unit) = max * 2^F
    assert_eq!(max().try_div(unit()), Err(OverflowDetected::TierOverflow));
}

#[test]
fn edges_that_fit_are_exact() {
    // the negative extreme is reachable by every operation that lands on it
    assert_eq!(min().try_mul(one()).unwrap(), min());
    assert_eq!(min().try_div(one()).unwrap(), min());
    let half_min = min().try_mul(FixedPoint::from_str("0.5")).unwrap();
    assert_eq!(half_min.try_add(half_min).unwrap(), min());
    assert_eq!(max().try_neg().unwrap().try_sub(unit()).unwrap(), min());
    assert_eq!(max().try_mul(one()).unwrap(), max());
    assert_eq!(max().try_div(one()).unwrap(), max());
}

#[test]
fn operators_match_their_twins_in_range() {
    let u = unit();
    let mut values = vec![one(), u, u.try_neg().unwrap(), max(), min(), top_power(), FixedPoint::ZERO];
    // literals the split can hold (3.25 is out of range at F=30)
    values.extend(["1.5", "-0.75", "3.25"].iter().filter_map(|s| FixedPoint::try_from_str(s).ok()));
    for &a in &values {
        for &b in &values {
            if let Ok(s) = a.try_add(b) { assert_eq!(a + b, s); }
            if let Ok(d) = a.try_sub(b) { assert_eq!(a - b, d); }
            if let Ok(p) = a.try_mul(b) { assert_eq!(a * b, p); }
            if let Ok(q) = a.try_div(b) { assert_eq!(a / b, q); }
        }
        if let Ok(n) = a.try_neg() { assert_eq!(-a, n); }
    }
}

#[test]
#[should_panic(expected = "overflow")]
fn add_panics_at_the_top() { let _ = max() + unit(); }

#[test]
#[should_panic(expected = "overflow")]
fn sub_panics_at_the_bottom() { let _ = min() - unit(); }

#[test]
#[should_panic(expected = "overflow")]
fn neg_panics_at_the_minimum() { let _ = -min(); }

#[test]
#[should_panic(expected = "overflow")]
fn mul_panics_out_of_range() { let _ = max() * max(); }

#[test]
#[should_panic(expected = "division by zero")]
fn div_panics_on_zero() { let _ = one() / FixedPoint::ZERO; }

#[test]
#[should_panic(expected = "overflow")]
fn div_panics_out_of_range() { let _ = max() / unit(); }

#[test]
#[should_panic(expected = "overflow")]
fn add_assign_panics_at_the_top() { let mut x = max(); x += unit(); }

#[test]
#[should_panic(expected = "overflow")]
fn vector_add_inherits_the_check() {
    let _ = FixedVector::from_slice(&[one(), max()]) + FixedVector::from_slice(&[one(), unit()]);
}

#[test]
#[should_panic(expected = "overflow")]
fn matrix_add_inherits_the_check() {
    let _ = FixedMatrix::from_slice(1, 2, &[one(), max()]) + FixedMatrix::from_slice(1, 2, &[one(), unit()]);
}
