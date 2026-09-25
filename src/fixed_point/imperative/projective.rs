//! L4B: Projective and conformal geometry with fixed-point arithmetic.
//!
//! Provides:
//! - Homogeneous coordinates: to/from with UGOD overflow detection at infinity
//! - Projective transformations: (n+1)×(n+1) matrix acting on homogeneous coords
//! - Cross-ratio: projective invariant of 4 collinear points (exact, one rounding)
//! - Stereographic projection: S^n ↔ R^n conformal mapping
//! - Möbius transformations: 2D (az+b)/(cz+d), real and complex
//!
//! **Compute tier:** sums of products (a transformed row, a Möbius numerator
//! or denominator, a composed coefficient) are formed exactly at the compute
//! tier and every result is rounded to storage once (nearest, ties toward
//! +infinity). Quotients divide the exact numerator by the exact denominator,
//! so a small nonzero denominator is never rounded to zero first.
//! Dehomogenization reports a point at infinity (w exactly zero) as
//! `DomainError` and a quotient beyond storage as `TierOverflow`.

use super::FixedPoint;
use super::FixedVector;
use super::FixedMatrix;
use super::linalg::{exact_dot, round_to_storage};
use super::wide_acc::{acc, divide_to_storage_nearest, narrow_triple_nearest, widen_product, widen_storage, Wide};
use crate::fixed_point::universal::fasc::stack_evaluator::{BinaryStorage, ComputeStorage};
use crate::fixed_point::universal::fasc::stack_evaluator::compute::{
    compute_add, compute_checked_add, compute_is_zero, compute_negate, compute_subtract, make_compute_int,
};
use crate::fixed_point::core_types::errors::OverflowDetected;
use crate::fixed_point::frac_config;
#[cfg(table_format = "q32_32")]
use crate::fixed_point::I256;
#[cfg(table_format = "q64_64")]
use crate::fixed_point::I512;
#[cfg(table_format = "q128_128")]
use crate::fixed_point::I1024;
#[cfg(table_format = "q256_256")]
use crate::fixed_point::I2048;

// ============================================================================
// Exact helpers
// ============================================================================

/// `sum a_i b_i - sum c_j d_j` over pairs of storage values, exact at the
/// compute tier (`2F` fractional bits). A sum beyond it is a `TierOverflow`.
fn exact_signed_sum(
    plus: &[(BinaryStorage, BinaryStorage)],
    minus: &[(BinaryStorage, BinaryStorage)],
) -> Result<ComputeStorage, OverflowDetected> {
    let dot = |t: &[(BinaryStorage, BinaryStorage)]| {
        let (a, b): (Vec<BinaryStorage>, Vec<BinaryStorage>) = t.iter().copied().unzip();
        exact_dot(&a, &b)
    };
    compute_checked_add(dot(plus)?, compute_negate(dot(minus)?))
}

/// A compute raw (`2F` fractional bits) as an exact `3F` value, the numerator
/// scale of [`divide_to_storage_nearest`] over a `2F` denominator.
#[inline]
fn lift_to_triple(c: ComputeStorage) -> acc::Orient {
    widen_product(c, widen_storage(FixedPoint::one().raw()))
}

/// `num / den` of two compute raws at the same scale, rounded once to storage.
fn quotient(num: ComputeStorage, den: ComputeStorage) -> Result<FixedPoint, OverflowDetected> {
    if compute_is_zero(&den) {
        return Err(OverflowDetected::DomainError);
    }
    Ok(FixedPoint::from_raw(divide_to_storage_nearest(lift_to_triple(num), den)?))
}

/// The accumulator's one.
#[inline]
fn orient_unit() -> acc::Orient {
    #[cfg(table_format = "q16_16")]
    { 1i128 }
    #[cfg(table_format = "q32_32")]
    { I256::from_i128(1) }
    #[cfg(table_format = "q64_64")]
    { I512::from_i128(1) }
    #[cfg(table_format = "q128_128")]
    { I1024::from_i128(1) }
    #[cfg(table_format = "q256_256")]
    { I2048::from_i128(1) }
}

/// Truncated quotient and remainder of two NON-NEGATIVE accumulator values.
#[inline]
fn orient_divmod(n: acc::Orient, d: acc::Orient) -> (acc::Orient, acc::Orient) {
    #[cfg(not(table_format = "q256_256"))]
    { (n / d, n % d) }
    #[cfg(table_format = "q256_256")]
    { crate::fixed_point::domains::binary_fixed::i2048::i2048_divmod(n, d) }
}

/// Exact product on the accumulator, `TierOverflow` when it does not fit
/// (instead of the width-budget panic of `Wide::mul_exact`).
fn orient_mul(x: acc::Orient, y: acc::Orient) -> Result<acc::Orient, OverflowDetected> {
    let magnitude = |v: acc::Orient| if Wide::is_negative(v) { -v } else { v };
    if Wide::bit_length(magnitude(x)) + Wide::bit_length(magnitude(y)) > <acc::Orient as Wide>::BITS - 1 {
        return Err(OverflowDetected::TierOverflow);
    }
    Ok(x.mul_exact(y))
}

/// `n / d` for two exact values at a common scale (any scale: it cancels),
/// rounded ONCE to storage, nearest with ties toward +infinity, checked.
///
/// Neither `n` nor `d` needs to fit the compute tier: the integer quotient is
/// taken on the accumulator, the `3F` fraction bits below it by restoring
/// division on the remainder, and the resulting exact floor at `3F` narrowed
/// by [`narrow_triple_nearest`] (floor plus the round bit is the nearest,
/// because a floor keeps every bit above the ones it discards).
fn ratio_to_storage(n: acc::Orient, d: acc::Orient) -> Result<BinaryStorage, OverflowDetected> {
    let zero = <acc::Orient as Wide>::zero();
    if d == zero {
        return Err(OverflowDetected::DivisionByZero);
    }
    let negative = Wide::is_negative(n) != Wide::is_negative(d) && n != zero;
    let na = if Wide::is_negative(n) { -n } else { n };
    let da = if Wide::is_negative(d) { -d } else { d };
    let (mut q, mut r) = orient_divmod(na, da);
    // a storage value has at most W = BITS / 4 bits; bounding the integer
    // part by it keeps q < 2^(W + 3F) inside the accumulator below
    if Wide::bit_length(q) > <acc::Orient as Wide>::BITS / 4 {
        return Err(OverflowDetected::TierOverflow);
    }
    let unit = orient_unit();
    for _ in 0..3 * frac_config::FRAC_BITS {
        // r < da, so 2r never leaves the accumulator
        let rest = da - r;
        if r >= rest {
            q = q + q + unit;
            r = r - rest;
        } else {
            q = q + q;
            r = r + r;
        }
    }
    // floor of the signed value at 3F: a negative quotient with a nonzero
    // remainder floors one unit further from zero
    let floor = if negative { -(if r != zero { q + unit } else { q }) } else { q };
    narrow_triple_nearest(floor)
}

// ============================================================================
// Homogeneous coordinates
// ============================================================================

/// Convert affine coordinates [x₁, ..., xₙ] to homogeneous [x₁, ..., xₙ, 1].
pub fn to_homogeneous(v: &FixedVector) -> FixedVector {
    let n = v.len();
    let mut h = FixedVector::new(n + 1);
    for i in 0..n {
        h[i] = v[i];
    }
    h[n] = FixedPoint::one();
    h
}

/// Convert homogeneous coordinates [x₁, ..., xₙ, w] to affine [x₁/w, ..., xₙ/w].
///
/// Returns `Err(OverflowDetected::DomainError)` if w ≈ 0 (point at infinity).
pub fn from_homogeneous(h: &FixedVector) -> Result<FixedVector, OverflowDetected> {
    let n = h.len();
    if n == 0 { return Err(OverflowDetected::DomainError); }
    let w = h[n - 1];
    if w.is_zero() {
        return Err(OverflowDetected::DomainError);
    }
    let mut v = FixedVector::new(n - 1);
    for i in 0..n - 1 {
        v[i] = h[i] / w;
    }
    Ok(v)
}

/// Check if a homogeneous point is at infinity (last component ≈ 0).
pub fn is_at_infinity(h: &FixedVector, tol: FixedPoint) -> bool {
    let n = h.len();
    if n == 0 { return true; }
    h[n - 1].abs() < tol
}

// ============================================================================
// Projective transformations
// ============================================================================

/// Apply a projective transformation H (n+1)×(n+1) to a point in affine coordinates.
///
/// Lifts to homogeneous, multiplies by H, then dehomogenizes. Each row of
/// `H [p; 1]` is an exact sum at the compute tier; each affine coordinate is
/// that row divided by the exact last row and rounded to storage once.
/// Returns `Err(DomainError)` if the transformed point is at infinity (w
/// exactly zero) and `Err(TierOverflow)` if a coordinate leaves storage.
pub fn projective_transform(
    h_matrix: &FixedMatrix,
    point: &FixedVector,
) -> Result<FixedVector, OverflowDetected> {
    let n = point.len();
    assert!(h_matrix.rows() == n + 1 && h_matrix.cols() == n + 1, "projective_transform: H must be (n+1)x(n+1)");
    let hp: Vec<BinaryStorage> = to_homogeneous(point).iter().map(|x| x.raw()).collect();
    let row = |i: usize| {
        let h: Vec<BinaryStorage> = (0..=n).map(|j| h_matrix.get(i, j).raw()).collect();
        exact_dot(&h, &hp)
    };
    let w = row(n)?;
    if compute_is_zero(&w) {
        return Err(OverflowDetected::DomainError);
    }
    let mut v = FixedVector::new(n);
    for i in 0..n {
        v[i] = quotient(row(i)?, w)?;
    }
    Ok(v)
}

/// Apply a projective transformation to a homogeneous-coordinate point.
///
/// Returns the result in homogeneous coordinates (no dehomogenization).
pub fn projective_transform_homogeneous(
    h_matrix: &FixedMatrix,
    point: &FixedVector,
) -> FixedVector {
    h_matrix.mul_vector(point)
}

/// Compose two projective transformations (matrix multiplication).
///
/// compose_projective(H₁, H₂) represents applying H₂ first, then H₁.
pub fn compose_projective(
    h1: &FixedMatrix,
    h2: &FixedMatrix,
) -> FixedMatrix {
    h1 * h2
}

// ============================================================================
// Cross-ratio
// ============================================================================

/// Cross-ratio of 4 collinear points (a, b, c, d) in R¹.
///
/// CR(a, b, c, d) = (a-c)(b-d) / ((a-d)(b-c))
///
/// This is the fundamental projective invariant: preserved under all
/// projective transformations. The differences and both products are exact
/// at the compute tier and above it; the quotient is rounded to storage once.
/// `Err(DomainError)` only when the denominator is exactly zero (two points
/// coincide), `Err(TierOverflow)` when the ratio leaves storage.
pub fn cross_ratio_1d(
    a: FixedPoint,
    b: FixedPoint,
    c: FixedPoint,
    d: FixedPoint,
) -> Result<FixedPoint, OverflowDetected> {
    // storage values sign-extended to the compute width: exact differences
    let (a, b, c, d) = (widen_storage(a.raw()), widen_storage(b.raw()), widen_storage(c.raw()), widen_storage(d.raw()));
    let numer = widen_product(compute_subtract(a, c), compute_subtract(b, d));
    let denom = widen_product(compute_subtract(a, d), compute_subtract(b, c));
    if denom == <acc::Orient as Wide>::zero() {
        return Err(OverflowDetected::DomainError);
    }
    Ok(FixedPoint::from_raw(ratio_to_storage(numer, denom)?))
}

/// Cross-ratio for 4 collinear points in R^n (using ratios of signed distances).
///
/// For collinear points, the cross-ratio is computed by projecting onto the line
/// direction and using 1D cross-ratio.
///
/// The coordinates along the line are the exact projections `(x - a) . dir`
/// with `dir = b - a`: the signed distances times `|dir|`, a common scale that
/// cancels in the ratio, so no square root or division is needed before the
/// single rounding of the quotient.
///
/// Returns `Err(DomainError)` if `a == b` or the ratio's denominator is exactly
/// zero, `Err(TierOverflow)` if it leaves storage.
pub fn cross_ratio(
    a: &FixedVector,
    b: &FixedVector,
    c: &FixedVector,
    d: &FixedVector,
) -> Result<FixedPoint, OverflowDetected> {
    let n = a.len();
    assert!(b.len() == n && c.len() == n && d.len() == n, "cross_ratio: dimension mismatch");
    // exact differences at the compute width, exact projections at 2F
    let diff = |x: &FixedVector| -> Vec<ComputeStorage> {
        (0..n).map(|i| compute_subtract(widen_storage(x[i].raw()), widen_storage(a[i].raw()))).collect()
    };
    let dir = diff(b);
    let project = |x: &FixedVector| super::wide_acc::exact_dot_compute(&diff(x), &dir);
    let tb = project(b)?;
    if tb == <acc::Orient as Wide>::zero() {
        return Err(OverflowDetected::DomainError);
    }
    let (tc, td) = (project(c)?, project(d)?);

    // CR(0, tb, tc, td) = (0 - tc)(tb - td) / ((0 - td)(tb - tc))
    let numer = orient_mul(-tc, tb.add_exact(-td)?)?;
    let denom = orient_mul(-td, tb.add_exact(-tc)?)?;
    if denom == <acc::Orient as Wide>::zero() {
        return Err(OverflowDetected::DomainError);
    }
    Ok(FixedPoint::from_raw(ratio_to_storage(numer, denom)?))
}

// ============================================================================
// Stereographic projection
// ============================================================================

/// Stereographic projection from S^n to R^n (north pole projection).
///
/// Projects a point on the unit sphere in R^{n+1} to R^n.
/// The north pole (0, ..., 0, 1) maps to infinity.
///
/// Formula: x_i = p_i / (1 - p_{n})  for i = 0..n-1
/// where p = (p₀, ..., p_n) is the sphere point with p_n as the "north" coordinate.
///
/// Returns Err if the point is at the north pole (p_n = 1).
pub fn stereo_project(p: &FixedVector) -> Result<FixedVector, OverflowDetected> {
    let n = p.len();
    if n < 2 { return Err(OverflowDetected::DomainError); }

    let pn = p[n - 1];
    let denom = FixedPoint::one() - pn;
    if denom.is_zero() {
        return Err(OverflowDetected::DomainError); // north pole
    }

    let mut x = FixedVector::new(n - 1);
    for i in 0..n - 1 {
        x[i] = p[i] / denom;
    }
    Ok(x)
}

/// Inverse stereographic projection from R^n to S^n.
///
/// Maps a point in R^n to the unit sphere in R^{n+1}.
///
/// Formula: p_i = 2x_i / (1 + |x|²)  for i = 0..n-1
///          p_n = (|x|² - 1) / (|x|² + 1)
///
/// `|x|²` and both numerators are exact at the compute tier; each component
/// is one division rounded to storage once.
///
/// Panics if `|x|²` exceeds the compute tier.
pub fn stereo_unproject(x: &FixedVector) -> FixedVector {
    const OVERFLOW: &str = "stereo_unproject: |x|^2 exceeds the compute tier";
    let n = x.len();
    let one = make_compute_int(1);

    let x_raw: Vec<BinaryStorage> = (0..n).map(|i| x[i].raw()).collect();
    let x_sq = exact_dot(&x_raw, &x_raw).expect(OVERFLOW);
    let denom = compute_add(one, x_sq); // 1 + |x|², exact

    // every component lies in [-1, 1]: the divisions cannot leave storage
    let mut p = FixedVector::new(n + 1);
    for i in 0..n {
        // 2 x_i at 3F: the compute-tier 2 times the storage x_i, exact
        let numer = widen_product(make_compute_int(2), widen_storage(x[i].raw()));
        p[i] = FixedPoint::from_raw(divide_to_storage_nearest(numer, denom).expect(OVERFLOW));
    }
    p[n] = quotient(compute_subtract(x_sq, one), denom).expect(OVERFLOW);
    p
}

// ============================================================================
// Möbius transformations (2D — complex plane)
// ============================================================================

/// A Möbius transformation on the complex plane: z ↦ (az+b)/(cz+d).
///
/// Represented by 4 FixedPoint values (a, b, c, d) treated as real.
/// For full complex Möbius, use `MoebiusComplex`.
///
/// The transformation preserves the cross-ratio and maps circles/lines to
/// circles/lines.
#[derive(Clone, Copy, Debug)]
pub struct Moebius {
    pub a: FixedPoint,
    pub b: FixedPoint,
    pub c: FixedPoint,
    pub d: FixedPoint,
}

impl Moebius {
    /// Create a new Möbius transformation.
    pub fn new(a: FixedPoint, b: FixedPoint, c: FixedPoint, d: FixedPoint) -> Self {
        Self { a, b, c, d }
    }

    /// Identity transformation: z ↦ z.
    pub fn identity() -> Self {
        Self {
            a: FixedPoint::one(),
            b: FixedPoint::ZERO,
            c: FixedPoint::ZERO,
            d: FixedPoint::one(),
        }
    }

    /// Apply the transformation to a real value: (ax+b)/(cx+d).
    ///
    /// Numerator and denominator are exact at the compute tier; the quotient
    /// is rounded to storage once. `Err(DomainError)` if `cx + d` is exactly
    /// zero, `Err(TierOverflow)` if the result leaves storage.
    pub fn apply(&self, x: FixedPoint) -> Result<FixedPoint, OverflowDetected> {
        let one = FixedPoint::one().raw();
        let x = x.raw();
        let numer = exact_signed_sum(&[(self.a.raw(), x), (self.b.raw(), one)], &[])?;
        let denom = exact_signed_sum(&[(self.c.raw(), x), (self.d.raw(), one)], &[])?;
        quotient(numer, denom)
    }

    /// Compose two Möbius transformations: (self ∘ other)(z) = self(other(z)).
    ///
    /// Composition corresponds to matrix multiplication:
    ///   [[a,b],[c,d]] * [[a',b'],[c',d']]
    /// Each coefficient is an exact pair sum at the compute tier, rounded once.
    ///
    /// Panics if a coefficient exceeds storage.
    pub fn compose(&self, other: &Moebius) -> Moebius {
        let pair = |p: FixedPoint, q: FixedPoint, r: FixedPoint, s: FixedPoint| {
            let sum = exact_signed_sum(&[(p.raw(), q.raw()), (r.raw(), s.raw())], &[])
                .expect("Moebius::compose: coefficient exceeds the compute tier");
            FixedPoint::from_raw(round_to_storage(sum))
        };
        Moebius {
            a: pair(self.a, other.a, self.b, other.c),
            b: pair(self.a, other.b, self.b, other.d),
            c: pair(self.c, other.a, self.d, other.c),
            d: pair(self.c, other.b, self.d, other.d),
        }
    }

    /// Inverse transformation: z ↦ (dz-b)/(-cz+a).
    pub fn inverse(&self) -> Moebius {
        // det = ad - bc
        Moebius {
            a: self.d,
            b: -self.b,
            c: -self.c,
            d: self.a,
        }
    }

    /// Determinant: ad - bc. Non-zero for valid Möbius transformation.
    ///
    /// The exact difference at the compute tier, rounded once. Panics if it
    /// exceeds storage.
    pub fn determinant(&self) -> FixedPoint {
        let det = exact_signed_sum(&[(self.a.raw(), self.d.raw())], &[(self.b.raw(), self.c.raw())])
            .expect("Moebius::determinant exceeds the compute tier");
        FixedPoint::from_raw(round_to_storage(det))
    }

    /// Convert to the corresponding 2×2 projective matrix [[a,b],[c,d]].
    pub fn to_matrix(&self) -> FixedMatrix {
        FixedMatrix::from_slice(2, 2, &[self.a, self.b, self.c, self.d])
    }
}

/// A Möbius transformation with complex coefficients: z ↦ (az+b)/(cz+d)
/// where a,b,c,d,z are complex numbers represented as (real, imag) pairs.
#[derive(Clone, Copy, Debug)]
pub struct MoebiusComplex {
    pub a: (FixedPoint, FixedPoint), // (real, imag)
    pub b: (FixedPoint, FixedPoint),
    pub c: (FixedPoint, FixedPoint),
    pub d: (FixedPoint, FixedPoint),
}

type Complex = (FixedPoint, FixedPoint);

/// `p q + r s` for complex storage values, exact at the compute tier
/// (`(re, im)` compute raws).
fn complex_mul_add_exact(p: Complex, q: Complex, r: Complex, s: Complex) -> Result<(ComputeStorage, ComputeStorage), OverflowDetected> {
    let re = exact_signed_sum(
        &[(p.0.raw(), q.0.raw()), (r.0.raw(), s.0.raw())],
        &[(p.1.raw(), q.1.raw()), (r.1.raw(), s.1.raw())],
    )?;
    let im = exact_signed_sum(
        &[(p.0.raw(), q.1.raw()), (p.1.raw(), q.0.raw()), (r.0.raw(), s.1.raw()), (r.1.raw(), s.0.raw())],
        &[],
    )?;
    Ok((re, im))
}

/// Complex divide of exact compute raws:
/// (a+bi)/(c+di) = ((ac+bd) + (bc-ad)i) / (c²+d²), with both numerators and
/// `c²+d²` exact on the accumulator and each part rounded to storage once.
fn complex_div(
    a: (ComputeStorage, ComputeStorage),
    b: (ComputeStorage, ComputeStorage),
) -> Result<(FixedPoint, FixedPoint), OverflowDetected> {
    let denom = widen_product(b.0, b.0).add_exact(widen_product(b.1, b.1))?;
    if denom == <acc::Orient as Wide>::zero() {
        return Err(OverflowDetected::DomainError);
    }
    let re = widen_product(a.0, b.0).add_exact(widen_product(a.1, b.1))?;
    let im = widen_product(a.1, b.0).add_exact(-widen_product(a.0, b.1))?;
    Ok((
        FixedPoint::from_raw(ratio_to_storage(re, denom)?),
        FixedPoint::from_raw(ratio_to_storage(im, denom)?),
    ))
}

impl MoebiusComplex {
    /// Create a new complex Möbius transformation.
    pub fn new(
        a: (FixedPoint, FixedPoint),
        b: (FixedPoint, FixedPoint),
        c: (FixedPoint, FixedPoint),
        d: (FixedPoint, FixedPoint),
    ) -> Self {
        Self { a, b, c, d }
    }

    /// Apply to a complex number z = (re, im).
    ///
    /// `az + b` and `cz + d` are exact at the compute tier, `|cz + d|²` and
    /// the products of the division exact above it; each part of the result
    /// is rounded to storage once. `Err(DomainError)` if `cz + d` is exactly
    /// zero, `Err(TierOverflow)` if the result leaves storage.
    pub fn apply(
        &self,
        z: (FixedPoint, FixedPoint),
    ) -> Result<(FixedPoint, FixedPoint), OverflowDetected> {
        let one = (FixedPoint::one(), FixedPoint::ZERO);
        let numer = complex_mul_add_exact(self.a, z, self.b, one)?;
        let denom = complex_mul_add_exact(self.c, z, self.d, one)?;
        complex_div(numer, denom)
    }

    /// Compose two complex Möbius transformations.
    ///
    /// Each coefficient part is an exact sum of four products at the compute
    /// tier, rounded once. Panics if a coefficient exceeds storage.
    pub fn compose(&self, other: &MoebiusComplex) -> MoebiusComplex {
        let term = |p: Complex, q: Complex, r: Complex, s: Complex| {
            let (re, im) = complex_mul_add_exact(p, q, r, s)
                .expect("MoebiusComplex::compose: coefficient exceeds the compute tier");
            (FixedPoint::from_raw(round_to_storage(re)), FixedPoint::from_raw(round_to_storage(im)))
        };
        MoebiusComplex {
            a: term(self.a, other.a, self.b, other.c),
            b: term(self.a, other.b, self.b, other.d),
            c: term(self.c, other.a, self.d, other.c),
            d: term(self.c, other.b, self.d, other.d),
        }
    }

    /// Inverse transformation.
    pub fn inverse(&self) -> MoebiusComplex {
        MoebiusComplex {
            a: self.d,
            b: (-self.b.0, -self.b.1),
            c: (-self.c.0, -self.c.1),
            d: self.a,
        }
    }
}
