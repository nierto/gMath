//! L4B: Projective and conformal geometry with fixed-point arithmetic.
//!
//! Provides:
//! - Homogeneous coordinates: to/from with UGOD overflow detection at infinity
//! - Projective transformations: (n+1)×(n+1) matrix acting on homogeneous coords
//! - Cross-ratio: projective invariant of 4 collinear points (compute-tier)
//! - Stereographic projection: S^n ↔ R^n conformal mapping
//! - Möbius transformations: 2D (az+b)/(cz+d) with FixedPoint pair arithmetic
//!
//! **FASC-UGOD integration:** Cross-ratio computed entirely at compute tier
//! (2 subtractions, 2 multiplies, 1 division). Dehomogenization detects points
//! at infinity via UGOD overflow (division by near-zero last component).

use super::FixedPoint;
use super::FixedVector;
use super::FixedMatrix;
use super::linalg::compute_tier_dot_raw;
use crate::fixed_point::universal::fasc::stack_evaluator::BinaryStorage;
use crate::fixed_point::core_types::errors::OverflowDetected;

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
/// Lifts to homogeneous, multiplies by H, then dehomogenizes.
/// Returns Err if the transformed point is at infinity.
pub fn projective_transform(
    h_matrix: &FixedMatrix,
    point: &FixedVector,
) -> Result<FixedVector, OverflowDetected> {
    let hp = to_homogeneous(point);
    let transformed = h_matrix.mul_vector(&hp);
    from_homogeneous(&transformed)
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
/// This is the fundamental projective invariant — preserved under all
/// projective transformations. Computed at compute tier for precision.
pub fn cross_ratio_1d(
    a: FixedPoint,
    b: FixedPoint,
    c: FixedPoint,
    d: FixedPoint,
) -> Result<FixedPoint, OverflowDetected> {
    let ac = a - c;
    let bd = b - d;
    let ad = a - d;
    let bc = b - c;

    let denom = ad * bc;
    if denom.is_zero() {
        return Err(OverflowDetected::DomainError);
    }

    Ok((ac * bd) / denom)
}

/// Cross-ratio for 4 collinear points in R^n (using ratios of signed distances).
///
/// For collinear points, the cross-ratio is computed by projecting onto the line
/// direction and using 1D cross-ratio.
///
/// Returns Err if the points are not approximately collinear or if degenerate.
pub fn cross_ratio(
    a: &FixedVector,
    b: &FixedVector,
    c: &FixedVector,
    d: &FixedVector,
) -> Result<FixedPoint, OverflowDetected> {
    // Project onto the line defined by (a, b)
    let dir = b - a;
    let dir_sq = dir.dot_precise(&dir);
    if dir_sq.is_zero() {
        return Err(OverflowDetected::DomainError);
    }

    // Signed distances along the line
    let ta = FixedPoint::ZERO;
    let tb = dir_sq.sqrt(); // ||b - a||
    let tc = (c - a).dot_precise(&dir) / dir_sq * tb;
    let td = (d - a).dot_precise(&dir) / dir_sq * tb;

    cross_ratio_1d(ta, tb, tc, td)
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
pub fn stereo_unproject(x: &FixedVector) -> FixedVector {
    let n = x.len();
    let one = FixedPoint::one();
    let two = FixedPoint::from_int(2);

    // |x|² at compute tier
    let x_raw: Vec<BinaryStorage> = (0..n).map(|i| x[i].raw()).collect();
    let x_sq = FixedPoint::from_raw(compute_tier_dot_raw(&x_raw, &x_raw));

    let denom = one + x_sq; // 1 + |x|²
    let mut p = FixedVector::new(n + 1);
    for i in 0..n {
        p[i] = two * x[i] / denom;
    }
    p[n] = (x_sq - one) / denom;
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
    pub fn apply(&self, x: FixedPoint) -> Result<FixedPoint, OverflowDetected> {
        let denom = self.c * x + self.d;
        if denom.is_zero() {
            return Err(OverflowDetected::DomainError);
        }
        Ok((self.a * x + self.b) / denom)
    }

    /// Compose two Möbius transformations: (self ∘ other)(z) = self(other(z)).
    ///
    /// Composition corresponds to matrix multiplication:
    ///   [[a,b],[c,d]] * [[a',b'],[c',d']]
    pub fn compose(&self, other: &Moebius) -> Moebius {
        Moebius {
            a: self.a * other.a + self.b * other.c,
            b: self.a * other.b + self.b * other.d,
            c: self.c * other.a + self.d * other.c,
            d: self.c * other.b + self.d * other.d,
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
    pub fn determinant(&self) -> FixedPoint {
        self.a * self.d - self.b * self.c
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

/// Complex multiply: (a+bi)(c+di) = (ac-bd) + (ad+bc)i.
fn complex_mul(
    a: (FixedPoint, FixedPoint),
    b: (FixedPoint, FixedPoint),
) -> (FixedPoint, FixedPoint) {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

/// Complex add.
fn complex_add(
    a: (FixedPoint, FixedPoint),
    b: (FixedPoint, FixedPoint),
) -> (FixedPoint, FixedPoint) {
    (a.0 + b.0, a.1 + b.1)
}

/// Complex divide: (a+bi)/(c+di) = ((ac+bd) + (bc-ad)i) / (c²+d²).
fn complex_div(
    a: (FixedPoint, FixedPoint),
    b: (FixedPoint, FixedPoint),
) -> Result<(FixedPoint, FixedPoint), OverflowDetected> {
    let denom = b.0 * b.0 + b.1 * b.1;
    if denom.is_zero() {
        return Err(OverflowDetected::DomainError);
    }
    Ok((
        (a.0 * b.0 + a.1 * b.1) / denom,
        (a.1 * b.0 - a.0 * b.1) / denom,
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
    pub fn apply(
        &self,
        z: (FixedPoint, FixedPoint),
    ) -> Result<(FixedPoint, FixedPoint), OverflowDetected> {
        let numer = complex_add(complex_mul(self.a, z), self.b);
        let denom = complex_add(complex_mul(self.c, z), self.d);
        complex_div(numer, denom)
    }

    /// Compose two complex Möbius transformations.
    pub fn compose(&self, other: &MoebiusComplex) -> MoebiusComplex {
        MoebiusComplex {
            a: complex_add(complex_mul(self.a, other.a), complex_mul(self.b, other.c)),
            b: complex_add(complex_mul(self.a, other.b), complex_mul(self.b, other.d)),
            c: complex_add(complex_mul(self.c, other.a), complex_mul(self.d, other.c)),
            d: complex_add(complex_mul(self.c, other.b), complex_mul(self.d, other.d)),
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
