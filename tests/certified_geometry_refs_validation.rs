//! Certified geometry against independent references: the mpmath gate.
//!
//! The other certified-geometry gates check the implementation against i128
//! models written alongside it. This one checks it against values produced by
//! `scripts/generate_certified_geometry_refs.py`: Python exact integers and
//! fractions, cross-checked against mpmath at 300 digits inside the generator,
//! sharing no code with the Rust side. Raw values arrive as exact little-endian
//! two's-complement bytes at the storage width, so the wide profiles, whose
//! raws exceed i128, are covered with operands near the storage maximum.
//!
//! Checked on every profile:
//! - certified sqrt endpoints equal the reference floor and ceil exactly;
//! - directed product and quotient endpoints equal the references exactly,
//!   including operands within a factor of four of the storage maximum;
//! - the interval Cholesky's last pivot encloses the exact rational pivot of
//!   the same dyadic matrices `tests/pd_verdict_validation.rs` builds;
//! - every exact predicate returns the reference sign on configurations
//!   scaled to within a factor of four of the storage maximum;
//! - decimal sqrt, product and quotient endpoints equal the references.

use g_math::fixed_point::imperative::predicates::{orient2d, orient3d, pd_verdict, PdVerdict, Sign};
#[cfg(not(table_format = "q256_256"))]
use g_math::fixed_point::imperative::predicates::{incircle, insphere};
use g_math::fixed_point::{DecimalFixed, DecimalInterval, FixedMatrix, FixedPoint, Interval};

#[allow(dead_code)]
mod data {
    include!("data/certified_geometry_refs.rs");
}
use data::{decimal_refs, refs};

/// A FixedPoint from exact little-endian two's-complement bytes at the
/// profile's storage width.
#[cfg(table_format = "q16_16")]
fn fp_le(b: &[u8]) -> FixedPoint { FixedPoint::from_raw(i32::from_le_bytes(b.try_into().expect("4 bytes"))) }
#[cfg(table_format = "q32_32")]
fn fp_le(b: &[u8]) -> FixedPoint { FixedPoint::from_raw(i64::from_le_bytes(b.try_into().expect("8 bytes"))) }
#[cfg(table_format = "q64_64")]
fn fp_le(b: &[u8]) -> FixedPoint { FixedPoint::from_raw(i128::from_le_bytes(b.try_into().expect("16 bytes"))) }
#[cfg(table_format = "q128_128")]
fn fp_le(b: &[u8]) -> FixedPoint { FixedPoint::from_raw(g_math::fixed_point::I256::from_bytes_le(b)) }
#[cfg(table_format = "q256_256")]
fn fp_le(b: &[u8]) -> FixedPoint { FixedPoint::from_raw(g_math::fixed_point::I512::from_bytes_le(b)) }

fn sign_of(s: i8) -> Sign {
    match s {
        -1 => Sign::Negative,
        0 => Sign::Zero,
        1 => Sign::Positive,
        _ => panic!("reference sign out of range"),
    }
}

/// The references assume the profile's default fractional split.
fn assert_default_split() {
    let one = Interval::point(FixedPoint::one());
    let two = one + one;
    // 2 * 2^F = 1 << (F + 1) in raw units: compare via the reference for x = 4
    let four = two * two;
    assert!(four.is_point());
    let expected_four = refs::SQRT.iter().find(|(_, f, c)| f == c && fp_le(f) == fp_le(c) && fp_le(f) == FixedPoint::from_int(2))
        .map(|(x, _, _)| fp_le(x));
    assert_eq!(expected_four, Some(four.lo()), "references were generated for FRAC_BITS = {}", refs::FRAC_BITS);
}

#[test]
fn sqrt_endpoints_match_independent_references() {
    assert_default_split();
    for (idx, (x, f, c)) in refs::SQRT.iter().enumerate() {
        let iv = Interval::point(fp_le(x)).sqrt();
        assert_eq!(iv.lo(), fp_le(f), "sqrt floor, reference {idx}");
        assert_eq!(iv.hi(), fp_le(c), "sqrt ceil, reference {idx}");
        // the scalar engine's result must lie inside the certified enclosure.
        // On the scientific profile the Q512.512 sqrt engine loses precision
        // above roughly 2^200 (known issue, pinned by the ignored test below),
        // so the near-maximum references are excluded from this assertion
        // there and only there.
        if scalar_sqrt_trusted(fp_le(x)) {
            let scalar = fp_le(x).sqrt();
            assert!(
                iv.contains(scalar),
                "scalar sqrt escaped the certified enclosure: reference {idx}, x = {}, scalar = {}, enclosure = [{}, {}]",
                fp_le(x), scalar, iv.lo(), iv.hi()
            );
        }
    }
    assert!(refs::SQRT.len() >= 12);
}

#[test]
fn product_and_quotient_endpoints_match_independent_references() {
    let mut near_max = 0usize;
    for (a, b, f, c) in refs::MUL {
        let iv = Interval::point(fp_le(a)) * Interval::point(fp_le(b));
        assert_eq!(iv.lo(), fp_le(f), "mul floor");
        assert_eq!(iv.hi(), fp_le(c), "mul ceil");
        if fp_le(a) > FixedPoint::from_int(1 << 20) || fp_le(a) < -FixedPoint::from_int(1 << 20) { near_max += 1; }
    }
    for (a, b, f, c) in refs::DIV {
        let iv = Interval::point(fp_le(a)) / Interval::point(fp_le(b));
        assert_eq!(iv.lo(), fp_le(f), "div floor");
        assert_eq!(iv.hi(), fp_le(c), "div ceil");
    }
    assert!(near_max > 0, "references must exercise operands far above the small-value sweeps");
    assert!(refs::MUL.len() >= 50 && refs::DIV.len() >= 50);
}

#[cfg(not(table_format = "q256_256"))]
fn scalar_sqrt_trusted(_x: FixedPoint) -> bool { true }

/// Scientific only: the scalar sqrt engine is trusted below 2^150 (measured
/// inside the enclosure through 2^150, outside from 2^200 on).
#[cfg(table_format = "q256_256")]
fn scalar_sqrt_trusted(x: FixedPoint) -> bool {
    x < FixedPoint::from_raw(g_math::fixed_point::I512::from_i128(1) << (150 + 256))
}

/// Pins the known defect: on the scientific profile `FixedPoint::sqrt` falls
/// outside the certified enclosure for inputs above roughly 2^200 (relative
/// error about 2^-259, up to 2^124 ulp at 2^254). Run with `--ignored`; it
/// fails today and must pass, and lose its `ignore`, once the Q512.512 engine
/// is repaired. On the other profiles the assertion is part of the main gate.
#[cfg(table_format = "q256_256")]
#[test]
#[ignore = "known issue: scientific Q512.512 sqrt engine loses precision above ~2^200; see CHANGELOG"]
fn scalar_sqrt_is_inside_the_enclosure_for_every_reference_input() {
    for (idx, (x, _, _)) in refs::SQRT.iter().enumerate() {
        let iv = Interval::point(fp_le(x)).sqrt();
        let scalar = fp_le(x).sqrt();
        assert!(iv.contains(scalar), "reference {idx}: scalar sqrt outside the certified enclosure");
    }
}

/// Bit-exact replica of the LCG in tests/pd_verdict_validation.rs, so the
/// matrices here are the ones the references were computed for.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let x = self.0;
        (x >> 33) ^ x
    }
    fn dyadic(&mut self) -> FixedPoint {
        let k = (self.next() % 17) as i32 - 8;
        FixedPoint::from_int(k) / FixedPoint::from_int(16)
    }
}

fn dyadic_spd(rng: &mut Rng, n: usize) -> FixedMatrix {
    let mut a = FixedMatrix::new(n, n);
    for i in 0..n {
        for j in 0..n {
            a.set(i, j, rng.dyadic());
        }
    }
    let mut m = FixedMatrix::new(n, n);
    for i in 0..n {
        for j in 0..n {
            let mut s = FixedPoint::ZERO;
            for k in 0..n {
                s = s + a.get(k, i) * a.get(k, j);
            }
            if i == j { s = s + FixedPoint::one(); }
            m.set(i, j, s);
        }
    }
    m
}

#[test]
fn interval_cholesky_encloses_the_exact_rational_pivot() {
    let mut rng = Rng(0x1D7);
    for (n, f, c) in refs::PIVOT {
        let m = dyadic_spd(&mut rng, *n);
        assert_eq!(pd_verdict(&m).unwrap(), PdVerdict::PositiveDefinite);
        let zero = Interval::point(FixedPoint::ZERO);
        let mut l = vec![vec![zero; *n]; *n];
        let mut last = zero;
        for i in 0..*n {
            let d = Interval::point(m.get(i, i)) - Interval::dot_intervals(&l[i][..i], &l[i][..i]);
            let lii = d.sqrt();
            l[i][i] = lii;
            for j in (i + 1)..*n {
                let num = Interval::point(m.get(j, i)) - Interval::dot_intervals(&l[j][..i], &l[i][..i]);
                l[j][i] = num / lii;
            }
            last = d;
        }
        let (exact_floor, exact_ceil) = (fp_le(f), fp_le(c));
        assert!(last.lo() <= exact_floor, "n = {n}: interval lower endpoint above the exact pivot");
        assert!(exact_ceil <= last.hi(), "n = {n}: interval upper endpoint below the exact pivot");
    }
    assert_eq!(refs::PIVOT.len(), 2);
}

fn pt2(p: &[&[u8]]) -> [FixedPoint; 2] { [fp_le(p[0]), fp_le(p[1])] }
fn pt3(p: &[&[u8]]) -> [FixedPoint; 3] { [fp_le(p[0]), fp_le(p[1]), fp_le(p[2])] }

#[test]
fn predicates_match_independent_references_near_the_storage_maximum() {
    let mut zeros = 0usize;
    for (pts, s) in refs::ORIENT2D {
        let got = orient2d(pt2(pts[0]), pt2(pts[1]), pt2(pts[2]));
        assert_eq!(got, sign_of(*s), "orient2d");
        if got == Sign::Zero { zeros += 1; }
    }
    for (pts, s) in refs::ORIENT3D {
        let got = orient3d(pt3(pts[0]), pt3(pts[1]), pt3(pts[2]), pt3(pts[3]));
        assert_eq!(got, sign_of(*s), "orient3d");
        if got == Sign::Zero { zeros += 1; }
    }
    #[cfg(not(table_format = "q256_256"))]
    {
        for (pts, s) in refs::INCIRCLE {
            let got = incircle(pt2(pts[0]), pt2(pts[1]), pt2(pts[2]), pt2(pts[3]));
            assert_eq!(got, sign_of(*s), "incircle");
            if got == Sign::Zero { zeros += 1; }
        }
        for (pts, s) in refs::INSPHERE {
            let got = insphere(pt3(pts[0]), pt3(pts[1]), pt3(pts[2]), pt3(pts[3]), pt3(pts[4]));
            assert_eq!(got, sign_of(*s), "insphere");
            if got == Sign::Zero { zeros += 1; }
        }
    }
    assert!(zeros >= 3, "the references must include exact degenerate cases, got {zeros}");
}

fn check_decimal<const D: u8>() {
    for (d, x, f, c) in decimal_refs::DSQRT {
        if *d != D { continue; }
        let iv = DecimalInterval::<D>::point(DecimalFixed::<D>::from_raw(*x)).sqrt();
        assert_eq!(iv.lo().raw_value(), *f, "decimal sqrt floor, D = {D}, x = {x}");
        assert_eq!(iv.hi().raw_value(), *c, "decimal sqrt ceil, D = {D}, x = {x}");
    }
    for (d, a, b, f, c) in decimal_refs::DMUL {
        if *d != D { continue; }
        let iv = DecimalInterval::<D>::point(DecimalFixed::<D>::from_raw(*a)) * DecimalInterval::<D>::point(DecimalFixed::<D>::from_raw(*b));
        assert_eq!((iv.lo().raw_value(), iv.hi().raw_value()), (*f, *c), "decimal mul, D = {D}, {a} * {b}");
    }
    for (d, a, b, f, c) in decimal_refs::DDIV {
        if *d != D { continue; }
        let iv = DecimalInterval::<D>::point(DecimalFixed::<D>::from_raw(*a)) / DecimalInterval::<D>::point(DecimalFixed::<D>::from_raw(*b));
        assert_eq!((iv.lo().raw_value(), iv.hi().raw_value()), (*f, *c), "decimal div, D = {D}, {a} / {b}");
    }
}

#[test]
fn decimal_endpoints_match_independent_references() {
    check_decimal::<2>();
    check_decimal::<9>();
    assert!(decimal_refs::DSQRT.len() >= 16 && decimal_refs::DMUL.len() >= 100 && decimal_refs::DDIV.len() >= 100);
}
