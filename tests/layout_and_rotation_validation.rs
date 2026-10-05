//! 0.6.5: the `FixedPoint` layout guarantee (zero-copy raw slice views) and
//! `FixedVector::rotate_pairs`. Values stay small so the suite holds on every
//! profile and realtime split it runs at.

use g_math::fixed_point::imperative::BinaryStorage;
use g_math::fixed_point::{FixedPoint, FixedVector};

fn fp(s: &str) -> FixedPoint {
    if let Some(rest) = s.strip_prefix('-') { -FixedPoint::from_str(rest) } else { FixedPoint::from_str(s) }
}

#[test]
fn fixed_point_has_the_layout_of_its_raw_storage() {
    assert_eq!(std::mem::size_of::<FixedPoint>(), std::mem::size_of::<BinaryStorage>());
    assert_eq!(std::mem::align_of::<FixedPoint>(), std::mem::align_of::<BinaryStorage>());
}

#[test]
fn raw_slice_views_are_the_same_memory() {
    let mut values = vec![fp("1.5"), fp("-0.25"), FixedPoint::ZERO, fp("0.75")];
    let raws: Vec<BinaryStorage> = values.iter().map(|v| v.raw()).collect();
    {
        let view = FixedPoint::raw_slice(&values);
        assert_eq!(view, &raws[..]);
        assert_eq!(view.as_ptr() as usize, values.as_ptr() as usize);
        assert_eq!(view.len(), values.len());
    }
    let back = FixedPoint::from_raw_slice(&raws);
    assert_eq!(back, &values[..]);
    assert_eq!(back.as_ptr() as usize, raws.as_ptr() as usize);
    // empty slices
    assert!(FixedPoint::raw_slice(&[]).is_empty());
    assert!(FixedPoint::from_raw_slice(&[]).is_empty());
    // writes through the mutable views land in the original
    FixedPoint::raw_slice_mut(&mut values)[2] = fp("0.5").raw();
    assert_eq!(values[2], fp("0.5"));
    let mut raws = raws;
    FixedPoint::from_raw_slice_mut(&mut raws)[0] = fp("-1");
    assert_eq!(raws[0], fp("-1").raw());
    // and through a vector
    let mut v = FixedVector::from_slice(&values);
    v.as_mut_slice()[1] = fp("0.125");
    assert_eq!(v[1], fp("0.125"));
    assert_eq!(FixedPoint::raw_slice(v.as_slice())[1], fp("0.125").raw());
}

/// `rotate_pairs` rotates the first `rotary_dim` elements and no others.
/// Every value here is a multiple of 1/2 below 2 with products that are
/// multiples of 1/4, so each output is exact at every split (2 to 30 fraction
/// bits) and the operator expression gives the same number. The one-rounding
/// accuracy is gated against exact references in `one_rounding_validation`.
#[test]
fn rotate_pairs_rotates_the_rotary_prefix() {
    // sin/cos of unrelated angles: the method must not assume sin^2 + cos^2 = 1
    let sin = [fp("0.5"), fp("-0.5"), fp("0"), fp("0.5")];
    let cos = [fp("0.5"), fp("1"), fp("-0.5"), fp("1")];
    let x: Vec<FixedPoint> = ["0.5", "-1", "1", "0.5", "-0.5", "0.5", "1", "-1", "0.5", "1"].iter().map(|s| fp(s)).collect();

    for &rotary_dim in &[0usize, 2, 4, 6, 8] {
        let half = rotary_dim / 2;
        let mut v = FixedVector::from_slice(&x);
        v.rotate_pairs(&sin, &cos, rotary_dim);
        for i in 0..half {
            let (x0, x1) = (x[i], x[i + half]);
            assert_eq!(v[i], x0 * cos[i] - x1 * sin[i], "dim {rotary_dim} lo {i}");
            assert_eq!(v[i + half], x0 * sin[i] + x1 * cos[i], "dim {rotary_dim} hi {i}");
        }
        // everything from rotary_dim on is untouched (a partial rotation)
        for i in rotary_dim..x.len() {
            assert_eq!(v[i], x[i], "dim {rotary_dim} tail {i}");
        }
    }

    // an exact case: a quarter turn swaps and negates
    let mut v = FixedVector::from_slice(&[fp("1.5"), fp("-0.5")]);
    v.rotate_pairs(&[FixedPoint::one()], &[FixedPoint::ZERO], 2);
    assert_eq!(v[0], fp("0.5"));
    assert_eq!(v[1], fp("1.5"));
}

#[test]
#[should_panic(expected = "rotary_dim exceeds")]
fn rotate_pairs_refuses_a_rotary_dim_past_the_vector() {
    let mut v = FixedVector::from_slice(&[FixedPoint::ZERO; 2]);
    v.rotate_pairs(&[FixedPoint::ZERO; 4], &[FixedPoint::ZERO; 4], 4);
}

#[test]
#[should_panic(expected = "shorter than")]
fn rotate_pairs_refuses_short_tables() {
    let mut v = FixedVector::from_slice(&[FixedPoint::ZERO; 4]);
    v.rotate_pairs(&[FixedPoint::ZERO; 1], &[FixedPoint::ZERO; 2], 4);
}
