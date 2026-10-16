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

use g_math::fixed_point::imperative::matrix_functions::{matrix_exp, matrix_log, matrix_pow, matrix_sqrt};

/// exp(0) = I, sqrt(diag(4, 9)) = diag(2, 3), log(I) = 0, A^1 = A, A^-1 = inverse.
fn matrix_function_identities<S: Storage, const F: u32>()
where
    Fixed<S, F>: g_math::fixed_point::imperative::matrix_functions::MatrixFunctions<Storage = S>,
{
    let v = |n: i32| Fixed::<S, F>::from_int(n);
    let z = FixedMatrix::<Fixed<S, F>>::new(2, 2);
    assert_eq!(matrix_exp(&z).unwrap(), FixedMatrix::identity(2));
    let d = FixedMatrix::from_slice(2, 2, &[v(4), v(0), v(0), v(9)]);
    let s = matrix_sqrt(&d).unwrap();
    let ulp = Fixed::<S, F>::from_raw(S::one());
    for (i, j, want) in [(0, 0, v(2)), (1, 1, v(3)), (0, 1, v(0)), (1, 0, v(0))] {
        assert!((s.get(i, j) - want).abs() <= ulp * v(2), "sqrt at F={F}: ({i},{j}) = {:?}", s.get(i, j).raw());
    }
    assert_eq!(matrix_log(&FixedMatrix::<Fixed<S, F>>::identity(2)).unwrap(), FixedMatrix::new(2, 2));
    let a = FixedMatrix::from_slice(2, 2, &[v(2), v(1), v(0), v(3)]);
    assert_eq!(matrix_pow(&a, v(1)).unwrap(), a);
    let inv = matrix_pow(&a, v(-1)).unwrap();
    let half = Fixed::<S, F>::one() / v(2);
    let third = Fixed::<S, F>::one() / v(3);
    assert_eq!(inv, FixedMatrix::from_slice(2, 2, &[half, -(half * third), v(0), third]));
}

#[test]
fn matrix_functions_run_at_every_width() {
    matrix_function_identities::<i32, 10>();
    matrix_function_identities::<i32, 16>();
    matrix_function_identities::<i32, 24>();
    matrix_function_identities::<i64, 32>();
    matrix_function_identities::<i128, 64>();
    matrix_function_identities::<I256, 128>();
    matrix_function_identities::<I512, 256>();
}

// ---- geometry layers, dynamics and tensors at every width (phase 4a) --------

use g_math::fixed_point::imperative::curvature::{scalar_curvature, SphereMetric};
use g_math::fixed_point::imperative::lie_group::{LieGroup, SO3};
use g_math::fixed_point::imperative::manifold::{HyperbolicSpace, Manifold, Sphere};
use g_math::fixed_point::imperative::ode::{ode_fn, rk4_step};
use g_math::fixed_point::imperative::projective::Moebius;
use g_math::fixed_point::imperative::tensor::{contract, Tensor};
use g_math::fixed_point::Scalar;

/// A few identities that hold at every width: exp_map(p, 0) = p on the
/// sphere, distance(p, p) = 0 on the hyperboloid, log(exp(xi)) = xi on SO(3)
/// to two units, the scalar curvature of the unit sphere is 2 to a few
/// units, the identity Moebius map and the identity contraction are exact,
/// and one RK4 step of x' = 0 leaves x unchanged.
fn geometry_identities<S: Storage, const F: u32>()
where
    Fixed<S, F>: Scalar<Storage = S> + g_math::fixed_point::imperative::matrix_functions::MatrixFunctions,
{
    let v = |n: i32| Fixed::<S, F>::from_int(n);
    let q = |n: i32| dyadic::<S, F>(n); // n / 256
    let ulp = Fixed::<S, F>::from_raw(S::one());
    let units = |x: Fixed<S, F>, want: Fixed<S, F>| ((x - want).abs() / ulp).to_int();
    // sphere
    let s = Sphere { dim: 2 };
    let p = FixedVector::from_slice(&[v(0), v(0), v(1)]);
    let zero = FixedVector::<Fixed<S, F>>::new(3);
    assert_eq!(s.exp_map(&p, &zero).unwrap(), p, "sphere exp_map at F={F}");
    let e0 = FixedVector::from_slice(&[v(1), v(0), v(0)]);
    let d = s.distance(&p, &e0).unwrap();
    let pi_half = Fixed::<S, F>::from_raw(Fixed::<S, F>::downscale_to_storage(Fixed::<S, F>::pi_half_at_compute_tier()).unwrap());
    assert!(units(d, pi_half) <= 2, "sphere quarter turn at F={F}: {} units", units(d, pi_half));
    // hyperboloid
    let h = HyperbolicSpace { dim: 2 };
    let x = FixedVector::from_slice(&[(v(1) + q(64) * q(64) + q(32) * q(32)).sqrt(), q(64), q(32)]);
    assert_eq!(h.distance(&x, &x).unwrap(), v(0), "hyperbolic distance(p, p) at F={F}");
    // SO(3)
    let xi = FixedVector::from_slice(&[q(64), q(-32), q(96)]);
    let g = SO3.lie_exp(&xi).unwrap();
    let back = SO3.lie_log(&g).unwrap();
    for i in 0..3 {
        assert!(units(back[i], xi[i]) <= 2, "so3 log(exp) at F={F}: component {i} off by {} units", units(back[i], xi[i]));
    }
    // curvature of the unit sphere in (theta, phi): R = 2
    let r = scalar_curvature(&SphereMetric { radius: v(1) }, &FixedVector::from_slice(&[q(256) + q(64), q(32)])).unwrap();
    let tol = if F <= 12 { 64 } else { 8 };
    assert!(units(r, v(2)) <= tol, "sphere scalar curvature at F={F}: {} units", units(r, v(2)));
    // projective
    let m = Moebius::<Fixed<S, F>>::identity();
    assert_eq!(m.apply(q(100)).unwrap(), q(100), "moebius identity at F={F}");
    // tensors
    let id = Tensor::from_data(&[2, 2], &[v(1), v(0), v(0), v(1)]);
    let t = Tensor::from_data(&[2, 2], &[q(1), q(2), q(3), q(4)]);
    assert_eq!(contract(&id, &t, &[(1, 0)]).data(), t.data(), "identity contraction at F={F}");
    // one RK4 step of x' = 0
    let still = ode_fn(|_t: Fixed<S, F>, x: &FixedVector<Fixed<S, F>>| FixedVector::<Fixed<S, F>>::new(x.len()));
    let x0 = FixedVector::from_slice(&[q(5), q(-7)]);
    assert_eq!(rk4_step(&still, v(0), &x0, q(16)), x0, "rk4 of a constant at F={F}");
}

#[test]
fn geometry_runs_at_every_width() {
    geometry_identities::<i32, 10>();
    geometry_identities::<i32, 16>();
    geometry_identities::<i32, 24>();
    geometry_identities::<i64, 32>();
    geometry_identities::<i128, 64>();
    geometry_identities::<I256, 128>();
    geometry_identities::<I512, 256>();
}

// ---- fused kernels and the compute tier at every width (phase 4b) ----------

use g_math::fixed_point::compute_tier;
use g_math::fixed_point::imperative::fused;

/// Identities that hold at every width: sqrt_sum_sq([3, 4]) = 5 exactly,
/// the softmax of equal scores sums to one within n units, silu(0) = 0,
/// sigmoid_mul(x, 0) = x / 2 within a unit, the rms factor of ones with no
/// eps is one, the entropy of a one-hot is zero, dot_many agrees with dot,
/// the quadratic form with the identity is the squared norm, and exp(0) at
/// the compute tier is one.
fn fused_identities<S: Storage, const F: u32>()
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    let v = |n: i32| Fixed::<S, F>::from_int(n);
    let q = |n: i32| dyadic::<S, F>(n); // n / 256
    let ulp = Fixed::<S, F>::from_raw(S::one());
    let units = |x: Fixed<S, F>, want: Fixed<S, F>| ((x - want).abs() / ulp).to_int();
    assert_eq!(fused::sqrt_sum_sq(&[v(3), v(4)]), v(5), "sqrt_sum_sq at F={F}");
    assert_eq!(fused::euclidean_distance(&[v(1), v(2)], &[v(4), v(6)]), v(5), "euclidean_distance at F={F}");
    let soft = fused::softmax(&[q(10), q(10), q(10), q(10)]).unwrap();
    let sum = soft.iter().fold(v(0), |s, &x| s + x);
    assert!(units(sum, v(1)) <= 4, "softmax sum at F={F}: {} units", units(sum, v(1)));
    assert_eq!(fused::silu(v(0)), v(0), "silu(0) at F={F}");
    let half = fused::sigmoid_mul(q(200), v(0));
    assert!(units(half, q(100)) <= 1, "sigmoid_mul(x, 0) at F={F}: {} units", units(half, q(100)));
    assert_eq!(fused::rms_norm_factor(&[v(1), v(1), v(1), v(1)], v(0)).unwrap(), v(1), "rms factor at F={F}");
    assert_eq!(fused::entropy(&[v(1), v(0), v(0)]).unwrap(), v(0), "entropy of a one-hot at F={F}");
    let (qv, k) = ([q(3), q(-5), q(7)], [q(1), q(2), q(3), q(-4), q(5), q(-6)]);
    let many = fused::dot_many(&qv, &k, 3);
    assert_eq!(many, vec![fused::dot(&qv, &k[..3]), fused::dot(&qv, &k[3..])], "dot_many at F={F}");
    let vec = FixedVector::from_slice(&[q(3), q(-5), q(7)]);
    assert_eq!(fused::quadratic_form(&vec, &FixedMatrix::identity(3)), fused::dot(&qv, &qv), "quadratic form at F={F}");
    assert_eq!(compute_tier::exp::<Fixed<S, F>>(compute_tier::from_fixed(v(0))), compute_tier::one::<Fixed<S, F>>(), "compute_tier exp(0) at F={F}");
    assert_eq!(compute_tier::to_fixed::<Fixed<S, F>>(compute_tier::sqrt::<Fixed<S, F>>(compute_tier::from_fixed(v(9)))), v(3), "compute_tier sqrt(9) at F={F}");
}

#[test]
fn fused_kernels_run_at_every_width() {
    fused_identities::<i32, 10>();
    fused_identities::<i32, 16>();
    fused_identities::<i32, 24>();
    fused_identities::<i64, 32>();
    fused_identities::<i128, 64>();
    fused_identities::<I256, 128>();
    fused_identities::<I512, 256>();
}

// ---- the certified interval and the exact predicates at every width (phase 4c)

use g_math::fixed_point::imperative::predicates::{orient2d, orient3d, pd_verdict, PdVerdict, Sign};
use g_math::fixed_point::{CircleStorage, Interval};

/// Identities at every width: [1, 2] * [3, 4] = [3, 8] exactly, sqrt([4, 9])
/// = [2, 3], 1/3 encloses its nearest rounding with width one unit, a
/// counterclockwise triangle is positive and a collinear one zero, the
/// tetrahedron sign, and the identity is proven positive definite.
fn certified_identities<S: Storage, const F: u32>()
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    let v = |n: i32| Fixed::<S, F>::from_int(n);
    let ulp = Fixed::<S, F>::from_raw(S::one());
    assert_eq!(Interval::new(v(1), v(2)) * Interval::new(v(3), v(4)), Interval::new(v(3), v(8)), "interval product at F={F}");
    assert_eq!(Interval::new(v(4), v(9)).sqrt(), Interval::new(v(2), v(3)), "interval sqrt at F={F}");
    let third = Interval::point(v(1)) / Interval::point(v(3));
    assert_eq!(third.width(), ulp, "1/3 enclosure width at F={F}");
    assert!(third.contains(v(1) / v(3)), "1/3 enclosure at F={F}");
    assert_eq!(orient2d([v(0), v(0)], [v(1), v(0)], [v(0), v(1)]), Sign::Positive, "orient2d at F={F}");
    assert_eq!(orient2d([v(0), v(0)], [v(1), v(1)], [v(2), v(2)]), Sign::Zero, "collinear at F={F}");
    assert_eq!(orient3d([v(0), v(0), v(0)], [v(1), v(0), v(0)], [v(0), v(1), v(0)], [v(0), v(0), v(1)]), Sign::Negative, "orient3d at F={F}");
    assert_eq!(pd_verdict(&FixedMatrix::<Fixed<S, F>>::identity(3)).unwrap(), PdVerdict::PositiveDefinite, "pd_verdict at F={F}");
}

fn circle_identities<S: CircleStorage, const F: u32>()
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    use g_math::fixed_point::imperative::predicates::{incircle, insphere};
    let v = |n: i32| Fixed::<S, F>::from_int(n);
    // the unit circle through (1,0), (0,1), (-1,0): the origin inside, (2,0) outside, (0,-1) on it
    assert_eq!(incircle([v(1), v(0)], [v(0), v(1)], [v(-1), v(0)], [v(0), v(0)]), Sign::Positive, "incircle inside at F={F}");
    assert_eq!(incircle([v(1), v(0)], [v(0), v(1)], [v(-1), v(0)], [v(2), v(0)]), Sign::Negative, "incircle outside at F={F}");
    assert_eq!(incircle([v(1), v(0)], [v(0), v(1)], [v(-1), v(0)], [v(0), v(-1)]), Sign::Zero, "incircle on at F={F}");
    let s = insphere([v(1), v(0), v(0)], [v(0), v(1), v(0)], [v(0), v(0), v(1)], [v(-1), v(0), v(0)], [v(0), v(0), v(0)]);
    assert_ne!(s, Sign::Zero, "insphere at F={F}");
}

#[test]
fn certified_arithmetic_runs_at_every_width() {
    certified_identities::<i32, 10>();
    certified_identities::<i32, 16>();
    certified_identities::<i32, 24>();
    certified_identities::<i64, 32>();
    certified_identities::<i128, 64>();
    certified_identities::<I256, 128>();
    certified_identities::<I512, 256>();
    circle_identities::<i32, 10>();
    circle_identities::<i32, 24>();
    circle_identities::<i64, 32>();
    circle_identities::<i128, 64>();
    circle_identities::<I256, 128>();
}
