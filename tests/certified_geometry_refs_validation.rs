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
//! - the fused quadratic form's interval equals `[floor, ceil]` and its scalar
//!   the nearest of references computed on values with exact rationals;
//! - decimal sqrt, product and quotient endpoints equal the references.

use g_math::fixed_point::imperative::predicates::{orient2d, orient3d, pd_verdict, PdVerdict, Sign};
#[cfg(not(table_format = "q256_256"))]
use g_math::fixed_point::imperative::predicates::{incircle, insphere};
use g_math::fixed_point::imperative::fused;
use g_math::fixed_point::{DecimalFixed, DecimalInterval, FixedMatrix, FixedPoint, FixedVector, Interval};

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

/// Bit length of the magnitude of little-endian two's-complement bytes.
fn magnitude_bits(b: &[u8]) -> u32 {
    let negative = b[b.len() - 1] & 0x80 != 0;
    // one's complement of a negative is |x| - 1: bit length within one of |x|
    let mag: Vec<u8> = b.iter().map(|x| if negative { !x } else { *x }).collect();
    for (i, byte) in mag.iter().enumerate().rev() {
        if *byte != 0 { return i as u32 * 8 + (8 - byte.leading_zeros()); }
    }
    0
}

fn sign_of(s: i8) -> Sign {
    match s {
        -1 => Sign::Negative,
        0 => Sign::Zero,
        1 => Sign::Positive,
        _ => panic!("reference sign out of range"),
    }
}

/// The references match the build's split: the sqrt(1) = 1 entry decodes to
/// exactly 1.0 only when the table's FRAC_BITS is the build's (every profile,
/// and every realtime GMATH_FRAC_BITS from 2 to 30, has a table).
fn assert_default_split() {
    let one = FixedPoint::one();
    let found = refs::SQRT.iter().any(|(x, f, c)| fp_le(x) == one && fp_le(f) == one && fp_le(c) == one);
    assert!(found, "references were generated for FRAC_BITS = {}", refs::FRAC_BITS);
}

#[test]
fn sqrt_endpoints_match_independent_references() {
    assert_default_split();
    for (idx, (x, f, c)) in refs::SQRT.iter().enumerate() {
        let iv = Interval::point(fp_le(x)).sqrt();
        assert_eq!(iv.lo(), fp_le(f), "sqrt floor, reference {idx}");
        assert_eq!(iv.hi(), fp_le(c), "sqrt ceil, reference {idx}");
        // the scalar engine's result must lie inside the certified enclosure,
        // including for inputs near the storage maximum. On the scientific
        // profile this once failed above ~2^200 (the Q512.512 engine lost
        // precision for large inputs; fixed by normalising the input).
        let scalar = fp_le(x).sqrt();
        assert!(
            iv.contains(scalar),
            "scalar sqrt escaped the certified enclosure: reference {idx}, x = {}, scalar = {}, enclosure = [{}, {}]",
            fp_le(x), scalar, iv.lo(), iv.hi()
        );
    }
    assert!(refs::SQRT.len() >= 9);
}

#[test]
fn product_and_quotient_endpoints_match_independent_references() {
    assert_default_split();
    let mut near_max = 0usize;
    for (a, b, f, c) in refs::MUL {
        let iv = Interval::point(fp_le(a)) * Interval::point(fp_le(b));
        assert_eq!(iv.lo(), fp_le(f), "mul floor");
        assert_eq!(iv.hi(), fp_le(c), "mul ceil");
        // the generator's near-maximum operands: at least W - 2 bits of magnitude
        if magnitude_bits(a) >= refs::STORAGE_BITS - 2 { near_max += 1; }
    }
    for (a, b, f, c) in refs::DIV {
        let iv = Interval::point(fp_le(a)) / Interval::point(fp_le(b));
        assert_eq!(iv.lo(), fp_le(f), "div floor");
        assert_eq!(iv.hi(), fp_le(c), "div ceil");
    }
    assert!(near_max > 0, "references must exercise operands far above the small-value sweeps");
    assert!(refs::MUL.len() >= 50 && refs::DIV.len() >= 50);
}

/// The fused quadratic form against references computed on values with
/// exact rationals (no raw-integer shifts in common with the kernel): the
/// interval equals `[floor, ceil]` and the fused scalar equals the nearest
/// with ties toward +infinity, on constructed ties, dyadic and random
/// matrices, and operands near the storage maximum.
#[test]
fn quadratic_form_endpoints_and_nearest_match_independent_references() {
    assert_default_split();
    let mut near_max = 0usize;
    let mut rounded_up = 0usize;
    for (n, v, m, f, c, near) in refs::QF {
        let fv = FixedVector::from_slice(&v.iter().map(|b| fp_le(b)).collect::<Vec<_>>());
        let fm = FixedMatrix::from_slice(*n, *n, &m.iter().map(|b| fp_le(b)).collect::<Vec<_>>());
        let iv = Interval::quadratic_form(&fv, &fm);
        assert_eq!(iv.lo(), fp_le(f), "quadratic form floor, n = {n}");
        assert_eq!(iv.hi(), fp_le(c), "quadratic form ceil, n = {n}");
        let scalar = fused::quadratic_form(&fv, &fm);
        assert_eq!(scalar, fp_le(near), "fused quadratic form nearest, n = {n}");
        assert!(iv.contains(scalar));
        // the generator's near-maximum operands are about 2^k with
        // k = (W - 1 + 2F) / 3 - 2 (the largest whose form still fits storage)
        let k = (refs::STORAGE_BITS - 1 + 2 * refs::FRAC_BITS) / 3 - 2;
        if magnitude_bits(v[0]) >= k { near_max += 1; }
        if fp_le(near) == fp_le(c) && fp_le(c) != fp_le(f) { rounded_up += 1; }
    }
    assert!(near_max >= 4, "references must exercise operands near the storage maximum");
    assert!(rounded_up >= 2, "references must exercise inexact values rounded upward (the constructed ties)");
    assert!(refs::QF.len() >= 20);
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
    assert_default_split();
    let mut rng = Rng(0x1D7);
    for (n, f, c) in refs::PIVOT {
        let m = dyadic_spd(&mut rng, *n);
        // Below 16 fraction bits the factor can be too wide to prove PD (at
        // 10 bits the n = 50 pivot 35 straddles zero): a sound Inconclusive,
        // see pd_verdict_validation, past which the last pivot cannot be
        // formed. The verdict must still never be wrong.
        let verdict = pd_verdict(&m).unwrap();
        if refs::FRAC_BITS < 16 {
            assert!(matches!(verdict, PdVerdict::Inconclusive { .. } | PdVerdict::PositiveDefinite), "{verdict:?}");
            if verdict != PdVerdict::PositiveDefinite { continue; }
        } else {
            assert_eq!(verdict, PdVerdict::PositiveDefinite);
        }
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
    // the dyadic k/16 family needs 4 <= F and entries up to 13.5 in range
    let f = refs::FRAC_BITS;
    let expected = if refs::STORAGE_BITS == 32 && !(4..=27).contains(&f) { 0 } else { 2 };
    assert_eq!(refs::PIVOT.len(), expected);
}

fn pt2(p: &[&[u8]]) -> [FixedPoint; 2] { [fp_le(p[0]), fp_le(p[1])] }
fn pt3(p: &[&[u8]]) -> [FixedPoint; 3] { [fp_le(p[0]), fp_le(p[1]), fp_le(p[2])] }

#[test]
fn predicates_match_independent_references_near_the_storage_maximum() {
    assert_default_split();
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
