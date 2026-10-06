//! Every width in one build.
//!
//! `Fixed<S, F>` makes the width part of the type, so one test binary can
//! compute with all of them, whatever profile the build selected. Two
//! things are checked here that no per-profile suite can check:
//!
//! - the same generic code runs at every width and satisfies the exact
//!   identities of fixed-point arithmetic there;
//! - every width agrees with the widest one. The Q256.256 result of each
//!   function is rounded to the narrower format and compared with what the
//!   narrower format computed. Both are rounded once from a wider compute
//!   tier, so they may differ by the double rounding only: at most one unit.
//!
//! Integer arithmetic only. The accuracy of each width against mpmath is
//! the job of the per-width reference gates.

use g_math::fixed_point::{Fixed, FixedNum, Storage, I256, I512, Q256_256};

/// A storage raw as an exact Q256.256 integer.
trait AtQ256: Storage {
    fn at_q256(self, frac_bits: u32) -> I512;
}
impl AtQ256 for i32 {
    fn at_q256(self, f: u32) -> I512 { I512::from_i128(self as i128) << (256 - f) as usize }
}
impl AtQ256 for i64 {
    fn at_q256(self, f: u32) -> I512 { I512::from_i128(self as i128) << (256 - f) as usize }
}
impl AtQ256 for i128 {
    fn at_q256(self, f: u32) -> I512 { I512::from_i128(self) << (256 - f) as usize }
}
impl AtQ256 for I256 {
    fn at_q256(self, f: u32) -> I512 { I512::from_i256(self) << (256 - f) as usize }
}

/// `value / 256` in the format `Fixed<S, F>` (exact for `F >= 8`).
fn dyadic<S: Storage, const F: u32>(value_times_256: i32) -> Fixed<S, F>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    Fixed::<S, F>::from_raw(S::from_i64(value_times_256 as i64).shl_bits(F - 8))
}

type Unary<S, const F: u32> = fn(Fixed<S, F>) -> Fixed<S, F>;

/// The functions and the arguments (in 1/256) they are compared on.
fn cases<S: Storage, const F: u32>() -> Vec<(&'static str, Unary<S, F>, &'static [i32])>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    const ANY: &[i32] = &[-640, -300, -129, -37, 0, 1, 64, 192, 257, 320, 700];
    const POSITIVE: &[i32] = &[1, 3, 64, 192, 256, 257, 320, 700, 1000];
    const UNIT: &[i32] = &[-250, -192, -64, -1, 0, 1, 64, 128, 192, 250];
    const ABOVE_ONE: &[i32] = &[256, 257, 320, 700, 1000];
    vec![
        ("exp", |x| x.exp(), ANY),
        ("ln", |x| x.ln(), POSITIVE),
        ("sqrt", |x| x.sqrt(), POSITIVE),
        ("sin", |x| x.sin(), ANY),
        ("cos", |x| x.cos(), ANY),
        ("atan", |x| x.atan(), ANY),
        ("asin", |x| x.asin(), UNIT),
        ("acos", |x| x.acos(), UNIT),
        ("sinh", |x| x.sinh(), ANY),
        ("cosh", |x| x.cosh(), ANY),
        ("tanh", |x| x.tanh(), ANY),
        ("asinh", |x| x.asinh(), ANY),
        ("acosh", |x| x.acosh(), ABOVE_ONE),
        ("atanh", |x| x.atanh(), UNIT),
    ]
}

/// Compare one format against Q256.256 on every case; returns how many
/// results matched the rounded reference exactly and how many were one unit
/// away.
fn agree_with_q256<S: AtQ256, const F: u32>() -> (usize, usize)
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    let reference = cases::<I512, 256>();
    let narrow = cases::<S, F>();
    let one = I512::from_i128(1);
    let (mut exact, mut off_by_one) = (0, 0);
    for ((name, f_ref, args), (_, f, _)) in reference.iter().zip(narrow.iter()) {
        for &a in args.iter() {
            let wide: Q256_256 = f_ref(dyadic::<I512, 256>(a));
            let got = f(dyadic::<S, F>(a)).raw().at_q256(F);
            // the reference rounded to F fractional bits: nearest, ties toward +infinity
            let shift = (256 - F) as usize;
            let rounded = ((wide.raw() + (one << (shift - 1))) >> shift) << shift;
            let diff = if got >= rounded { got - rounded } else { rounded - got };
            let unit = one << shift;
            assert!(
                diff <= unit,
                "{name}({a}/256) at F={F} on a {}-bit word is more than one unit from Q256.256",
                S::BITS
            );
            if diff == I512::zero() { exact += 1 } else { off_by_one += 1 }
        }
    }
    (exact, off_by_one)
}

/// Identities that hold exactly in every fixed-point format.
fn exact_identities<S: Storage, const F: u32>()
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    let v = |n: i32| Fixed::<S, F>::from_int(n);
    let (zero, one) = (Fixed::<S, F>::ZERO, Fixed::<S, F>::one());
    assert_eq!(v(0), zero);
    assert_eq!(v(1), one);
    assert_eq!(v(3) * v(4), v(12));
    assert_eq!(v(-3) * v(4), v(-12));
    assert_eq!(v(12) / v(4), v(3));
    assert_eq!(v(7) - v(9), v(-2));
    assert_eq!(-v(5) + v(5), zero);
    assert_eq!(v(-6).abs(), v(6));
    assert_eq!(v(-6).to_int(), -6);
    assert!(v(-1).is_negative() && !one.is_negative() && zero.is_zero());
    assert!(v(-2) < v(-1) && v(-1) < zero && zero < one);
    // halves and quarters are exact, and so is their arithmetic
    let (half, quarter) = (one / v(2), one / v(4));
    assert_eq!(half + half, one);
    assert_eq!(half * half, quarter);
    assert_eq!(quarter * v(4), one);
    assert_eq!((v(3) * half).to_int(), 1); // floor(1.5)
    assert_eq!((-(v(3) * half)).to_int(), -2); // floor(-1.5)
    // the smallest step: one raw unit, and the tie rule (toward +infinity)
    let ulp = Fixed::<S, F>::from_raw(S::one());
    assert_eq!(ulp * half, ulp); // 0.5 units rounds up to 1
    assert_eq!((-ulp) * half, zero); // -0.5 units rounds up to 0
    assert_eq!(ulp / v(2), ulp);
    assert_eq!((-ulp) / v(2), zero);
    // exact values of the transcendentals
    assert_eq!(zero.exp(), one);
    assert_eq!(one.ln(), zero);
    assert_eq!(v(4).sqrt(), v(2));
    assert_eq!(quarter.sqrt(), half);
    assert_eq!(zero.sin(), zero);
    assert_eq!(zero.cos(), one);
    assert_eq!(zero.atan(), zero);
    assert_eq!(zero.tanh(), zero);
    assert_eq!(zero.sinh(), zero);
    assert_eq!(zero.cosh(), one);
    // fallible forms report instead of panicking
    assert!(zero.try_ln().is_err());
    assert!(v(-1).try_sqrt().is_err());
    assert!(v(2).try_asin().is_err());
    assert!(one.try_div(zero).is_err());
    let max = Fixed::<S, F>::from_raw(S::one().shl_bits(S::BITS - 2));
    assert!(max.try_add(max).is_err());
    assert!(max.try_mul(v(2)).is_err());
    // the float boundary is exact on values a float holds
    assert_eq!(Fixed::<S, F>::from_f64(0.75), half + quarter);
    assert_eq!((half + quarter).to_f64().to_bits(), 0.75f64.to_bits());
    assert_eq!(Fixed::<S, F>::from_f64(-2.5).to_f64().to_bits(), (-2.5f64).to_bits());
}

#[test]
fn the_same_code_is_exact_at_every_width() {
    exact_identities::<i32, 8>();
    exact_identities::<i32, 10>();
    exact_identities::<i32, 12>();
    exact_identities::<i32, 16>();
    exact_identities::<i32, 20>();
    exact_identities::<i32, 24>();
    exact_identities::<i64, 32>();
    exact_identities::<i128, 64>();
    exact_identities::<I256, 128>();
    exact_identities::<I512, 256>();
}

#[test]
fn every_width_agrees_with_q256() {
    let mut report = Vec::new();
    macro_rules! width {
        ($s:ty, $f:literal) => {{
            let (exact, off) = agree_with_q256::<$s, $f>();
            report.push(format!("{:>4}-bit F={:<3} exact {exact:>3}, one unit {off:>3}", <$s as g_math::fixed_point::WideInt>::BITS, $f));
        }};
    }
    width!(i32, 8);
    width!(i32, 10);
    width!(i32, 12);
    width!(i32, 16);
    width!(i32, 20);
    width!(i32, 24);
    width!(i64, 32);
    width!(i128, 64);
    width!(I256, 128);
    for line in &report {
        println!("{line}");
    }
}

#[test]
fn widths_coexist_in_one_expression_chain() {
    // Two crates in one binary would each name their own width; here one
    // function does. The values are independent types: mixing them is a
    // compile error, converting is explicit (through the raw integer).
    let a = Fixed::<i32, 10>::from_int(3).sqrt();
    let b = Fixed::<i128, 64>::from_int(3).sqrt();
    let b_at_10 = ((b.raw() + (1i128 << 53)) >> 54) as i32;
    assert!((a.raw() - b_at_10).abs() <= 1);
    assert_eq!(std::mem::size_of_val(&a), 4);
    assert_eq!(std::mem::size_of_val(&b), 16);
    assert_eq!(std::mem::size_of::<Q256_256>(), 64);
}

// ============================================================================
// Containers: the same generic code at every width
// ============================================================================

use g_math::fixed_point::{FixedMatrix, FixedVector};

/// Vector and matrix identities that hold exactly in every format.
fn container_identities<S: Storage, const F: u32>()
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    let v = |n: i32| Fixed::<S, F>::from_int(n);
    let a = FixedVector::from_slice(&[v(1), v(2), v(3)]);
    let b = FixedVector::from_slice(&[v(4), v(-5), v(6)]);
    assert_eq!(a.dot(&b), v(12));
    assert_eq!(a.length_squared(), v(14));
    assert_eq!(FixedVector::from_slice(&[v(3), v(4)]).length(), v(5));
    assert_eq!((&a + &b)[1], v(-3));
    assert_eq!((&a - &b)[2], v(-3));
    assert_eq!((-&a)[0], v(-1));
    assert_eq!((a.clone() * v(2))[2], v(6));
    assert_eq!(a.cross(&b), FixedVector::from_slice(&[v(27), v(6), v(-13)]));
    assert_eq!(a.metric_distance_safe(&FixedVector::from_slice(&[v(1), v(2), v(7)])), v(4));
    let mut unit = FixedVector::from_slice(&[v(0), v(-3), v(4)]);
    unit.normalize();
    let (e3, e4) = (v(3) / v(5), v(4) / v(5));
    assert_eq!(unit, FixedVector::from_slice(&[v(0), -e3, e4]));
    // rotation by a quarter turn, exact
    let mut r = FixedVector::from_slice(&[v(1), v(2), v(3), v(4)]);
    r.rotate_pairs(&[v(1), v(1)], &[v(0), v(0)], 4);
    assert_eq!(r, FixedVector::from_slice(&[v(-3), v(-4), v(1), v(2)]));

    let m = FixedMatrix::from_slice(2, 2, &[v(1), v(2), v(3), v(4)]);
    let n = FixedMatrix::from_slice(2, 2, &[v(0), v(1), v(-1), v(0)]);
    assert_eq!(&m * &n, FixedMatrix::from_slice(2, 2, &[v(-2), v(1), v(-4), v(3)]));
    assert_eq!(m.mul_vector(&FixedVector::from_slice(&[v(1), v(-1)])), FixedVector::from_slice(&[v(-1), v(-1)]));
    assert_eq!(m.transpose().get(0, 1), v(3));
    assert_eq!(m.trace(), v(5));
    assert_eq!((&m + &n).get(1, 0), v(2));
    assert_eq!((&m - &n).get(0, 1), v(1));
    assert_eq!((m.clone() * v(2)).get(1, 1), v(8));
    assert_eq!(FixedMatrix::<Fixed<S, F>>::identity(3).trace(), v(3));
    assert_eq!(a.outer_product(&b).get(2, 1), v(-15));
    assert_eq!(m.kronecker(&n).get(1, 2), v(-2));
    // a half-unit tie in a dot product rounds toward +infinity
    let ulp = Fixed::<S, F>::from_raw(S::one());
    let half = Fixed::<S, F>::one() / v(2);
    assert_eq!(FixedVector::from_slice(&[ulp]).dot(&FixedVector::from_slice(&[half])), ulp);
    assert_eq!(FixedVector::from_slice(&[-ulp]).dot(&FixedVector::from_slice(&[half])), Fixed::ZERO);
}

#[test]
fn containers_are_exact_at_every_width() {
    container_identities::<i32, 8>();
    container_identities::<i32, 10>();
    container_identities::<i32, 16>();
    container_identities::<i32, 24>();
    container_identities::<i64, 32>();
    container_identities::<i128, 64>();
    container_identities::<I256, 128>();
    container_identities::<I512, 256>();
}

// ============================================================================
// Decompositions and derived operations at every width
// ============================================================================

use g_math::fixed_point::imperative::decompose::{cholesky_decompose, eigen_symmetric, lu_decompose, qr_decompose, svd_decompose};
use g_math::fixed_point::imperative::derived::{determinant, inverse, solve};

fn decomposition_identities<S: Storage, const F: u32>()
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    let v = |n: i32| Fixed::<S, F>::from_int(n);
    let half = Fixed::<S, F>::one() / v(2);
    // a dyadic SPD matrix with an exact factorisation: [4 2; 2 5] = L L^T, L = [2 0; 1 2]
    let a = FixedMatrix::from_slice(2, 2, &[v(4), v(2), v(2), v(5)]);
    let chol = cholesky_decompose(&a).unwrap();
    assert_eq!(chol.l, FixedMatrix::from_slice(2, 2, &[v(2), v(0), v(1), v(2)]));
    let lu = lu_decompose(&a).unwrap();
    assert_eq!(lu.determinant(), v(16));
    assert_eq!(determinant(&a).unwrap(), v(16));
    let b = FixedVector::from_slice(&[v(8), v(9)]);
    let eighth = half * half * half;
    assert_eq!(solve(&a, &b).unwrap(), FixedVector::from_slice(&[v(1) + v(3) * eighth, v(1) + v(2) * eighth]));
    let inv = inverse(&a).unwrap();
    assert_eq!(inv, FixedMatrix::from_slice(2, 2, &[v(5) / v(16), -v(2) / v(16), -v(2) / v(16), v(4) / v(16)]));
    // QR of an orthogonal-friendly matrix: Q R = A within one unit, R upper triangular
    let m = FixedMatrix::from_slice(2, 2, &[v(3), v(1), v(4), v(2)]);
    let qr = qr_decompose(&m).unwrap();
    let back = &qr.q * &qr.r;
    let ulp = Fixed::<S, F>::from_raw(S::one());
    for i in 0..2 { for j in 0..2 { assert!((back.get(i, j) - m.get(i, j)).abs() <= ulp * v(2), "QR at F={F}"); } }
    assert_eq!(qr.r.get(1, 0), Fixed::ZERO);
    // symmetric eigenproblem with exact eigenvalues: [2 1; 1 2] -> 1, 3
    let s = FixedMatrix::from_slice(2, 2, &[v(2), v(1), v(1), v(2)]);
    let eig = eigen_symmetric(&s).unwrap();
    let mut vals = vec![eig.values[0], eig.values[1]];
    vals.sort();
    assert!((vals[0] - v(1)).abs() <= ulp * v(4) && (vals[1] - v(3)).abs() <= ulp * v(4), "eigen at F={F}");
    // singular values of diag(3, 4) are 4, 3
    let d = FixedMatrix::from_slice(2, 2, &[v(3), v(0), v(0), v(4)]);
    let svd = svd_decompose(&d).unwrap();
    assert!((svd.sigma[0] - v(4)).abs() <= ulp * v(4) && (svd.sigma[1] - v(3)).abs() <= ulp * v(4), "SVD at F={F}");
}

#[test]
fn decompositions_run_at_every_width() {
    decomposition_identities::<i32, 8>();
    decomposition_identities::<i32, 10>();
    decomposition_identities::<i32, 16>();
    decomposition_identities::<i32, 24>();
    decomposition_identities::<i64, 32>();
    decomposition_identities::<i128, 64>();
    decomposition_identities::<I256, 128>();
    decomposition_identities::<I512, 256>();
}
