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

use super::FixedVector;
use super::FixedMatrix;
use super::kernels;
use super::kernels::Orient;
use crate::fixed_point::fixed_num::{Compute, Fixed, FixedNum};
use crate::fixed_point::width::{Storage, WideInt};
use crate::fixed_point::core_types::errors::OverflowDetected;

// ============================================================================
// Exact helpers
// ============================================================================

/// `sum a_i b_i - sum c_j d_j` over pairs of storage values, exact at the
/// compute tier (`2F` fractional bits). A sum beyond it is a `TierOverflow`.
fn exact_signed_sum<S: Storage, const F: u32>(
    plus: &[(S, S)],
    minus: &[(S, S)],
) -> Result<Compute<Fixed<S, F>>, OverflowDetected>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    let dot = |t: &[(S, S)]| {
        let (a, b): (Vec<S>, Vec<S>) = t.iter().copied().unzip();
        kernels::exact_dot::<Fixed<S, F>>(&a, &b)
    };
    Fixed::<S, F>::compute_checked_add(dot(plus)?, Fixed::<S, F>::compute_negate(dot(minus)?))
}

/// A compute raw (`2F` fractional bits) as an exact `3F` value, the numerator
/// scale of [`divide_to_storage_nearest`] over a `2F` denominator.
#[inline]
fn lift_to_triple<S: Storage, const F: u32>(c: Compute<Fixed<S, F>>) -> Orient<Fixed<S, F>>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    S::widen_product(c, S::widen(Fixed::one().raw()))
}

/// `num / den` of two compute raws at the same scale, rounded once to storage.
fn quotient<S: Storage, const F: u32>(num: Compute<Fixed<S, F>>, den: Compute<Fixed<S, F>>) -> Result<Fixed<S, F>, OverflowDetected>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    if Fixed::<S, F>::compute_is_zero(&den) {
        return Err(OverflowDetected::DomainError);
    }
    Ok(Fixed::from_raw(kernels::divide_to_storage_nearest::<Fixed<S, F>>(lift_to_triple(num), den)?))
}

/// The accumulator's one.
#[inline]
fn orient_unit<S: Storage, const F: u32>() -> Orient<Fixed<S, F>>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    <Orient<Fixed<S, F>> as WideInt>::one()
}

/// Truncated quotient and remainder of two NON-NEGATIVE accumulator values.
#[inline]
fn orient_divmod<S: Storage, const F: u32>(n: Orient<Fixed<S, F>>, d: Orient<Fixed<S, F>>) -> (Orient<Fixed<S, F>>, Orient<Fixed<S, F>>)
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    n.divmod(d)
}

/// Exact product on the accumulator, `TierOverflow` when it does not fit
/// (instead of the width-budget panic of `WideInt::mul_exact`).
fn orient_mul<S: Storage, const F: u32>(x: Orient<Fixed<S, F>>, y: Orient<Fixed<S, F>>) -> Result<Orient<Fixed<S, F>>, OverflowDetected>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    let magnitude = |v: Orient<Fixed<S, F>>| if WideInt::is_negative(v) { -v } else { v };
    if WideInt::bit_length(magnitude(x)) + WideInt::bit_length(magnitude(y)) > <Orient<Fixed<S, F>> as WideInt>::BITS - 1 {
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
fn ratio_to_storage<S: Storage, const F: u32>(n: Orient<Fixed<S, F>>, d: Orient<Fixed<S, F>>) -> Result<S, OverflowDetected>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    let zero = <Orient<Fixed<S, F>> as WideInt>::zero();
    if d == zero {
        return Err(OverflowDetected::DivisionByZero);
    }
    let negative = WideInt::is_negative(n) != WideInt::is_negative(d) && n != zero;
    let na = if WideInt::is_negative(n) { -n } else { n };
    let da = if WideInt::is_negative(d) { -d } else { d };
    let (mut q, mut r) = orient_divmod(na, da);
    // a storage value has at most W = BITS / 4 bits; bounding the integer
    // part by it keeps q < 2^(W + 3F) inside the accumulator below
    if WideInt::bit_length(q) > <Orient<Fixed<S, F>> as WideInt>::BITS / 4 {
        return Err(OverflowDetected::TierOverflow);
    }
    let unit = orient_unit::<S, F>();
    for _ in 0..3 * F {
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
    kernels::narrow_triple_nearest::<Fixed<S, F>>(floor)
}

// ============================================================================
// Homogeneous coordinates
// ============================================================================

/// Convert affine coordinates [x₁, ..., xₙ] to homogeneous [x₁, ..., xₙ, 1].
pub fn to_homogeneous<S: Storage, const F: u32>(v: &FixedVector<Fixed<S, F>>) -> FixedVector<Fixed<S, F>>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    let n = v.len();
    let mut h = FixedVector::<Fixed<S, F>>::new(n + 1);
    for i in 0..n {
        h[i] = v[i];
    }
    h[n] = Fixed::one();
    h
}

/// Convert homogeneous coordinates [x₁, ..., xₙ, w] to affine [x₁/w, ..., xₙ/w].
///
/// Returns `Err(OverflowDetected::DomainError)` if w ≈ 0 (point at infinity).
pub fn from_homogeneous<S: Storage, const F: u32>(h: &FixedVector<Fixed<S, F>>) -> Result<FixedVector<Fixed<S, F>>, OverflowDetected>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    let n = h.len();
    if n == 0 { return Err(OverflowDetected::DomainError); }
    let w = h[n - 1];
    if w.is_zero() {
        return Err(OverflowDetected::DomainError);
    }
    let mut v = FixedVector::<Fixed<S, F>>::new(n - 1);
    for i in 0..n - 1 {
        v[i] = h[i] / w;
    }
    Ok(v)
}

/// Check if a homogeneous point is at infinity (last component ≈ 0).
pub fn is_at_infinity<S: Storage, const F: u32>(h: &FixedVector<Fixed<S, F>>, tol: Fixed<S, F>) -> bool
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
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
pub fn projective_transform<S: Storage, const F: u32>(
    h_matrix: &FixedMatrix<Fixed<S, F>>,
    point: &FixedVector<Fixed<S, F>>,
) -> Result<FixedVector<Fixed<S, F>>, OverflowDetected>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    let n = point.len();
    assert!(h_matrix.rows() == n + 1 && h_matrix.cols() == n + 1, "projective_transform: H must be (n+1)x(n+1)");
    let hp: Vec<S> = to_homogeneous(point).iter().map(|x| x.raw()).collect();
    let row = |i: usize| {
        let h: Vec<S> = (0..=n).map(|j| h_matrix.get(i, j).raw()).collect();
        kernels::exact_dot::<Fixed<S, F>>(&h, &hp)
    };
    let w = row(n)?;
    if Fixed::<S, F>::compute_is_zero(&w) {
        return Err(OverflowDetected::DomainError);
    }
    let mut v = FixedVector::<Fixed<S, F>>::new(n);
    for i in 0..n {
        v[i] = quotient(row(i)?, w)?;
    }
    Ok(v)
}

/// Apply a projective transformation to a homogeneous-coordinate point.
///
/// Returns the result in homogeneous coordinates (no dehomogenization).
pub fn projective_transform_homogeneous<S: Storage, const F: u32>(
    h_matrix: &FixedMatrix<Fixed<S, F>>,
    point: &FixedVector<Fixed<S, F>>,
) -> FixedVector<Fixed<S, F>>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    h_matrix.mul_vector(point)
}

/// Compose two projective transformations (matrix multiplication).
///
/// compose_projective(H₁, H₂) represents applying H₂ first, then H₁.
pub fn compose_projective<S: Storage, const F: u32>(
    h1: &FixedMatrix<Fixed<S, F>>,
    h2: &FixedMatrix<Fixed<S, F>>,
) -> FixedMatrix<Fixed<S, F>>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
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
pub fn cross_ratio_1d<S: Storage, const F: u32>(
    a: Fixed<S, F>,
    b: Fixed<S, F>,
    c: Fixed<S, F>,
    d: Fixed<S, F>,
) -> Result<Fixed<S, F>, OverflowDetected>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    // storage values sign-extended to the compute width: exact differences
    let (a, b, c, d) = (S::widen(a.raw()), S::widen(b.raw()), S::widen(c.raw()), S::widen(d.raw()));
    let numer = S::widen_product(Fixed::<S, F>::compute_subtract(a, c), Fixed::<S, F>::compute_subtract(b, d));
    let denom = S::widen_product(Fixed::<S, F>::compute_subtract(a, d), Fixed::<S, F>::compute_subtract(b, c));
    if denom == <Orient<Fixed<S, F>> as WideInt>::zero() {
        return Err(OverflowDetected::DomainError);
    }
    Ok(Fixed::from_raw(ratio_to_storage(numer, denom)?))
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
pub fn cross_ratio<S: Storage, const F: u32>(
    a: &FixedVector<Fixed<S, F>>,
    b: &FixedVector<Fixed<S, F>>,
    c: &FixedVector<Fixed<S, F>>,
    d: &FixedVector<Fixed<S, F>>,
) -> Result<Fixed<S, F>, OverflowDetected>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    let n = a.len();
    assert!(b.len() == n && c.len() == n && d.len() == n, "cross_ratio: dimension mismatch");
    // exact differences at the compute width, exact projections at 2F
    let diff = |x: &FixedVector<Fixed<S, F>>| -> Vec<Compute<Fixed<S, F>>> {
        (0..n).map(|i| Fixed::<S, F>::compute_subtract(S::widen(x[i].raw()), S::widen(a[i].raw()))).collect()
    };
    let dir = diff(b);
    let project = |x: &FixedVector<Fixed<S, F>>| kernels::exact_dot_compute::<Fixed<S, F>>(&diff(x), &dir);
    let tb = project(b)?;
    if tb == <Orient<Fixed<S, F>> as WideInt>::zero() {
        return Err(OverflowDetected::DomainError);
    }
    let (tc, td) = (project(c)?, project(d)?);

    // CR(0, tb, tc, td) = (0 - tc)(tb - td) / ((0 - td)(tb - tc))
    let numer = orient_mul(-tc, tb.add_exact(-td)?)?;
    let denom = orient_mul(-td, tb.add_exact(-tc)?)?;
    if denom == <Orient<Fixed<S, F>> as WideInt>::zero() {
        return Err(OverflowDetected::DomainError);
    }
    Ok(Fixed::from_raw(ratio_to_storage(numer, denom)?))
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
pub fn stereo_project<S: Storage, const F: u32>(p: &FixedVector<Fixed<S, F>>) -> Result<FixedVector<Fixed<S, F>>, OverflowDetected>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    let n = p.len();
    if n < 2 { return Err(OverflowDetected::DomainError); }

    let pn = p[n - 1];
    let denom = Fixed::one() - pn;
    if denom.is_zero() {
        return Err(OverflowDetected::DomainError); // north pole
    }

    let mut x = FixedVector::<Fixed<S, F>>::new(n - 1);
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
pub fn stereo_unproject<S: Storage, const F: u32>(x: &FixedVector<Fixed<S, F>>) -> FixedVector<Fixed<S, F>>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    const OVERFLOW: &str = "stereo_unproject: |x|^2 exceeds the compute tier";
    let n = x.len();
    let one = Fixed::<S, F>::make_compute_int(1);

    let x_raw: Vec<S> = (0..n).map(|i| x[i].raw()).collect();
    let x_sq = kernels::exact_dot::<Fixed<S, F>>(&x_raw, &x_raw).expect(OVERFLOW);
    let denom = Fixed::<S, F>::compute_add(one, x_sq); // 1 + |x|², exact

    // every component lies in [-1, 1]: the divisions cannot leave storage
    let mut p = FixedVector::<Fixed<S, F>>::new(n + 1);
    for i in 0..n {
        // 2 x_i at 3F: the compute-tier 2 times the storage x_i, exact
        let numer = S::widen_product(Fixed::<S, F>::make_compute_int(2), S::widen(x[i].raw()));
        p[i] = Fixed::from_raw(kernels::divide_to_storage_nearest::<Fixed<S, F>>(numer, denom).expect(OVERFLOW));
    }
    p[n] = quotient(Fixed::<S, F>::compute_subtract(x_sq, one), denom).expect(OVERFLOW);
    p
}

// ============================================================================
// Möbius transformations (2D — complex plane)
// ============================================================================

/// A Möbius transformation on the complex plane: z ↦ (az+b)/(cz+d).
///
/// Represented by 4 Fixed<S, F> values (a, b, c, d) treated as real.
/// For full complex Möbius, use `MoebiusComplex<Fixed<S, F>>`.
///
/// The transformation preserves the cross-ratio and maps circles/lines to
/// circles/lines.
#[derive(Clone, Copy, Debug)]
pub struct Moebius<T> {
    pub a: T,
    pub b: T,
    pub c: T,
    pub d: T,
}

impl<S: Storage, const F: u32> Moebius<Fixed<S, F>>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    /// Create a new Möbius transformation.
    pub fn new(a: Fixed<S, F>, b: Fixed<S, F>, c: Fixed<S, F>, d: Fixed<S, F>) -> Self {
        Self { a, b, c, d }
    }

    /// Identity transformation: z ↦ z.
    pub fn identity() -> Self {
        Self {
            a: Fixed::one(),
            b: Fixed::ZERO,
            c: Fixed::ZERO,
            d: Fixed::one(),
        }
    }

    /// Apply the transformation to a real value: (ax+b)/(cx+d).
    ///
    /// Numerator and denominator are exact at the compute tier; the quotient
    /// is rounded to storage once. `Err(DomainError)` if `cx + d` is exactly
    /// zero, `Err(TierOverflow)` if the result leaves storage.
    pub fn apply(&self, x: Fixed<S, F>) -> Result<Fixed<S, F>, OverflowDetected> {
        let one = Fixed::one().raw();
        let x = x.raw();
        let numer = exact_signed_sum(&[(self.a.raw(), x), (self.b.raw(), one)], &[])?;
        let denom = exact_signed_sum(&[(self.c.raw(), x), (self.d.raw(), one)], &[])?;
        quotient(numer, denom)
    }

    /// Compose two Möbius transformations: (self ∘ other)(z) = self(other(z)).
    ///
    /// Composition corresponds to matrix multiplication:
    ///   \[\[a,b\],\[c,d\]\] * \[\[a',b'\],\[c',d'\]\]
    /// Each coefficient is an exact pair sum at the compute tier, rounded once.
    ///
    /// Panics if a coefficient exceeds storage.
    pub fn compose(&self, other: &Moebius<Fixed<S, F>>) -> Moebius<Fixed<S, F>> {
        let pair = |p: Fixed<S, F>, q: Fixed<S, F>, r: Fixed<S, F>, s: Fixed<S, F>| {
            let sum = exact_signed_sum(&[(p.raw(), q.raw()), (r.raw(), s.raw())], &[])
                .expect("Moebius::compose: coefficient exceeds the compute tier");
            Fixed::from_raw(kernels::round_to_storage::<Fixed<S, F>>(sum))
        };
        Moebius {
            a: pair(self.a, other.a, self.b, other.c),
            b: pair(self.a, other.b, self.b, other.d),
            c: pair(self.c, other.a, self.d, other.c),
            d: pair(self.c, other.b, self.d, other.d),
        }
    }

    /// Inverse transformation: z ↦ (dz-b)/(-cz+a).
    pub fn inverse(&self) -> Moebius<Fixed<S, F>> {
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
    pub fn determinant(&self) -> Fixed<S, F> {
        let det = exact_signed_sum(&[(self.a.raw(), self.d.raw())], &[(self.b.raw(), self.c.raw())])
            .expect("Moebius::determinant exceeds the compute tier");
        Fixed::from_raw(kernels::round_to_storage::<Fixed<S, F>>(det))
    }

    /// Convert to the corresponding 2×2 projective matrix \[\[a,b\],\[c,d\]\].
    pub fn to_matrix(&self) -> FixedMatrix<Fixed<S, F>> {
        FixedMatrix::<Fixed<S, F>>::from_slice(2, 2, &[self.a, self.b, self.c, self.d])
    }
}

/// A Möbius transformation with complex coefficients: z ↦ (az+b)/(cz+d)
/// where a,b,c,d,z are complex numbers represented as (real, imag) pairs.
#[derive(Clone, Copy, Debug)]
pub struct MoebiusComplex<T> {
    pub a: (T, T), // (real, imag)
    pub b: (T, T),
    pub c: (T, T),
    pub d: (T, T),
}

type Complex<S, const F: u32> = (Fixed<S, F>, Fixed<S, F>);

/// `p q + r s` for complex storage values, exact at the compute tier
/// (`(re, im)` compute raws).
fn complex_mul_add_exact<S: Storage, const F: u32>(p: Complex<S, F>, q: Complex<S, F>, r: Complex<S, F>, s: Complex<S, F>) -> Result<(Compute<Fixed<S, F>>, Compute<Fixed<S, F>>), OverflowDetected>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
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
fn complex_div<S: Storage, const F: u32>(
    a: (Compute<Fixed<S, F>>, Compute<Fixed<S, F>>),
    b: (Compute<Fixed<S, F>>, Compute<Fixed<S, F>>),
) -> Result<(Fixed<S, F>, Fixed<S, F>), OverflowDetected>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    let denom = S::widen_product(b.0, b.0).add_exact(S::widen_product(b.1, b.1))?;
    if denom == <Orient<Fixed<S, F>> as WideInt>::zero() {
        return Err(OverflowDetected::DomainError);
    }
    let re = S::widen_product(a.0, b.0).add_exact(S::widen_product(a.1, b.1))?;
    let im = S::widen_product(a.1, b.0).add_exact(-S::widen_product(a.0, b.1))?;
    Ok((
        Fixed::from_raw(ratio_to_storage(re, denom)?),
        Fixed::from_raw(ratio_to_storage(im, denom)?),
    ))
}

impl<S: Storage, const F: u32> MoebiusComplex<Fixed<S, F>>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    /// Create a new complex Möbius transformation.
    pub fn new(
        a: (Fixed<S, F>, Fixed<S, F>),
        b: (Fixed<S, F>, Fixed<S, F>),
        c: (Fixed<S, F>, Fixed<S, F>),
        d: (Fixed<S, F>, Fixed<S, F>),
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
        z: (Fixed<S, F>, Fixed<S, F>),
    ) -> Result<(Fixed<S, F>, Fixed<S, F>), OverflowDetected> {
        let one = (Fixed::one(), Fixed::ZERO);
        let numer = complex_mul_add_exact(self.a, z, self.b, one)?;
        let denom = complex_mul_add_exact(self.c, z, self.d, one)?;
        complex_div(numer, denom)
    }

    /// Compose two complex Möbius transformations.
    ///
    /// Each coefficient part is an exact sum of four products at the compute
    /// tier, rounded once. Panics if a coefficient exceeds storage.
    pub fn compose(&self, other: &MoebiusComplex<Fixed<S, F>>) -> MoebiusComplex<Fixed<S, F>> {
        let term = |p: Complex<S, F>, q: Complex<S, F>, r: Complex<S, F>, s: Complex<S, F>| {
            let (re, im) = complex_mul_add_exact(p, q, r, s)
                .expect("MoebiusComplex::compose: coefficient exceeds the compute tier");
            (Fixed::from_raw(kernels::round_to_storage::<Fixed<S, F>>(re)), Fixed::from_raw(kernels::round_to_storage::<Fixed<S, F>>(im)))
        };
        MoebiusComplex {
            a: term(self.a, other.a, self.b, other.c),
            b: term(self.a, other.b, self.b, other.d),
            c: term(self.c, other.a, self.d, other.c),
            d: term(self.c, other.b, self.d, other.d),
        }
    }

    /// Inverse transformation.
    pub fn inverse(&self) -> MoebiusComplex<Fixed<S, F>> {
        MoebiusComplex {
            a: self.d,
            b: (-self.b.0, -self.b.1),
            c: (-self.c.0, -self.c.1),
            d: self.a,
        }
    }
}
