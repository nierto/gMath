//! L1D: Matrix functions: exp, log, sqrt, pow.
//!
//! All operations run on a wide matrix internally and round once per output
//! element at the very end. On every profile but realtime that is the compute
//! tier (`ComputeMatrix`, `2 x FRAC_BITS` fractional bits); on realtime, whose
//! compute tier is only `2F` bits in an i64, it is a Q64.64 matrix
//! (`WideMatrix`), 40 to 56 guard bits beyond the compute tier across the
//! gated splits (0.6.3 ran realtime at `2F`: `expm` of a norm-7 matrix was 8
//! units off at 8 fraction bits, `logm` 6).
//!
//! Internal `_compute` variants accept and return ComputeMatrix, enabling
//! chains like `matrix_pow` (log → scalar_mul → exp) to stay wide
//! throughout: the matrix analog of FASC's BinaryCompute chain persistence.

use super::compute_matrix::ComputeMatrix;
use super::decompose::lu_decompose;
use super::kernels;
use super::wide_matrix::WideMatrix;
use super::FixedMatrix;
use crate::fixed_point::core_types::errors::OverflowDetected;
use crate::fixed_point::fixed_num::{Compute, Fixed, FixedNum};
use crate::fixed_point::width::Storage;
use crate::fixed_point::{I256, I512};

/// The matrix arithmetic the matrix functions run on.
trait FnMatrix<S: Storage, const F: u32>: Sized
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    type S: Copy;
    fn identity(n: usize) -> Self;
    fn dim(&self) -> usize;
    fn copy(&self) -> Self;
    fn add(&self, o: &Self) -> Result<Self, OverflowDetected>;
    fn sub(&self, o: &Self) -> Result<Self, OverflowDetected>;
    fn halve(&self) -> Self;
    fn mat_mul(&self, o: &Self) -> Result<Self, OverflowDetected>;
    fn scalar_mul(&self, s: Self::S) -> Result<Self, OverflowDetected>;
    /// `num / den` as a scalar, rounded once.
    fn fraction(num: i64, den: i64) -> Self::S;
    fn is_zero(&self) -> bool;
    /// `||M||_1` rounded to storage (step counts only).
    fn norm_1(&self) -> Fixed<S, F>;
    fn frobenius_below(&self, t: Fixed<S, F>) -> bool;
    /// `||self||_F < ||reference||_F * 2^-bits`, compared at the working
    /// precision (a threshold below one storage unit).
    fn step_below(&self, reference: &Self, bits: u32) -> bool;
    /// `D X = RHS` for the columns of `RHS`.
    fn solve(d: &Self, rhs: &Self) -> Result<Self, OverflowDetected>;
    fn inverse(&self) -> Result<Self, OverflowDetected>;
}

impl<S: Storage, const F: u32> FnMatrix<S, F> for ComputeMatrix<Fixed<S, F>>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    type S = Compute<Fixed<S, F>>;
    fn identity(n: usize) -> Self { ComputeMatrix::identity(n) }
    fn dim(&self) -> usize { self.rows() }
    fn copy(&self) -> Self { ComputeMatrix::copy(self) }
    fn add(&self, o: &Self) -> Result<Self, OverflowDetected> { Ok(ComputeMatrix::add(self, o)) }
    fn sub(&self, o: &Self) -> Result<Self, OverflowDetected> { Ok(ComputeMatrix::sub(self, o)) }
    fn halve(&self) -> Self { ComputeMatrix::halve(self) }
    fn mat_mul(&self, o: &Self) -> Result<Self, OverflowDetected> { Ok(ComputeMatrix::mat_mul(self, o)) }
    fn scalar_mul(&self, s: Self::S) -> Result<Self, OverflowDetected> { Ok(ComputeMatrix::scalar_mul(self, s)) }
    fn fraction(num: i64, den: i64) -> Self::S {
        Fixed::<S, F>::compute_mul_div_int(Fixed::<S, F>::make_compute_int(1), num, den).expect("coefficient fits the compute tier")
    }
    fn is_zero(&self) -> bool { self.frobenius_norm_compute().is_zero() }
    fn norm_1(&self) -> Fixed<S, F> { self.norm_1_compute() }
    fn frobenius_below(&self, t: Fixed<S, F>) -> bool { self.frobenius_norm_compute() < t }
    fn step_below(&self, reference: &Self, bits: u32) -> bool {
        let norm = |m: &Self| -> Option<Self::S> {
            let mut sum = Fixed::<S, F>::make_compute_int(0);
            for r in 0..m.rows() {
                for c in 0..m.cols() {
                    let v = m.get(r, c);
                    sum = Fixed::<S, F>::compute_checked_add(sum, Fixed::<S, F>::compute_checked_multiply(v, v).ok()?).ok()?;
                }
            }
            Some(Fixed::<S, F>::sqrt_at_compute_tier(sum))
        };
        match (norm(self), norm(reference)) {
            (Some(step), Some(r)) => step < kernels::compute_abs_shr::<Fixed<S, F>>(r, bits),
            _ => false,
        }
    }
    fn solve(d: &Self, rhs: &Self) -> Result<Self, OverflowDetected> {
        let lu = super::compute_matrix::compute_lu_decompose(d)?;
        let n = rhs.rows();
        let mut out = ComputeMatrix::new(n, rhs.cols());
        for j in 0..rhs.cols() {
            let x = lu.solve(&rhs.col_vec(j))?;
            for i in 0..n {
                out.set(i, j, x[i]);
            }
        }
        Ok(out)
    }
    fn inverse(&self) -> Result<Self, OverflowDetected> { super::compute_matrix::compute_lu_decompose(self)?.inverse() }
}

impl<const F: u32> FnMatrix<i32, F> for WideMatrix
where
    Fixed<i32, F>: FixedNum<Storage = i32>,
{
    type S = i128;
    fn identity(n: usize) -> Self { WideMatrix::identity(n) }
    fn dim(&self) -> usize { self.rows() }
    fn copy(&self) -> Self { self.clone() }
    fn add(&self, o: &Self) -> Result<Self, OverflowDetected> { WideMatrix::add(self, o) }
    fn sub(&self, o: &Self) -> Result<Self, OverflowDetected> { WideMatrix::sub(self, o) }
    fn halve(&self) -> Self { WideMatrix::halve(self) }
    fn mat_mul(&self, o: &Self) -> Result<Self, OverflowDetected> { WideMatrix::mat_mul(self, o) }
    fn scalar_mul(&self, s: i128) -> Result<Self, OverflowDetected> { WideMatrix::scalar_mul(self, s) }
    fn fraction(num: i64, den: i64) -> i128 { WideMatrix::fraction(num, den) }
    fn is_zero(&self) -> bool { WideMatrix::is_zero(self) }
    fn norm_1(&self) -> Fixed<i32, F> { WideMatrix::norm_1::<F>(self) }
    fn frobenius_below(&self, t: Fixed<i32, F>) -> bool { WideMatrix::frobenius_below::<F>(self, t) }
    fn step_below(&self, reference: &Self, bits: u32) -> bool { WideMatrix::step_below(self, reference, bits) }
    fn solve(d: &Self, rhs: &Self) -> Result<Self, OverflowDetected> {
        let lu = d.lu()?;
        let n = rhs.rows();
        let mut out = WideMatrix::new(n, n);
        for j in 0..n {
            let x = lu.solve(&(0..n).map(|r| rhs.get(r, j)).collect::<Vec<_>>())?;
            for i in 0..n {
                WideMatrix::set(&mut out, i, j, x[i]);
            }
        }
        Ok(out)
    }
    fn inverse(&self) -> Result<Self, OverflowDetected> { self.lu()?.inverse() }
}

// ============================================================================
// Padé [6/6] coefficients for matrix exponential
// ============================================================================

/// b_k = (12 - k)! 6! / (12! (6 - k)! k!) as exact fractions. Each is rounded
/// once at the working width (before 0.6.4 each was a 21-digit decimal rounded
/// to STORAGE: b_5 and b_6 were 0 at 10 fraction bits).
const PADE_B: [(i64, i64); 7] = [
    (1, 1),
    (1, 2),
    (5, 44),
    (1, 66),
    (1, 792),
    (1, 15840),
    (1, 665280),
];

// ============================================================================
// Matrix exponential: exp(A) via Padé [6/6] with scaling and squaring
// ============================================================================

fn exp_wide<S: Storage, const F: u32, M: FnMatrix<S, F>>(a: &M) -> Result<M, OverflowDetected>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    let n = a.dim();

    if a.is_zero() {
        return Ok(M::identity(n));
    }

    // Scaling: find s such that ||A||_1 / 2^s < 0.5 (only the count matters)
    let a_norm = a.norm_1();
    let mut s = 0u32;
    let mut scale = a_norm;
    let one = Fixed::<S, F>::one();
    let half = one.div_count(2);
    while scale >= half {
        scale = scale / (one + one);
        s += 1;
    }
    // ... then far enough below 0.5 for the Pade [6/6] truncation error,
    // 6!6!/(12!13!) ||B||^13 = 2^-42.4 2^-13k at ||B|| = 2^-k, amplified by
    // the 2^s of the squarings, to stay below 2^-(F + 6): 12k >= F - 36.4 + L
    // with L = log2 ||A||_1 = s - 1. k = 1 (the rule above) up to 32 fraction
    // bits; 3 on Q64.64, 8 on Q128.128, 19 on Q256.256, where ||B|| < 0.5
    // alone left 21 units (Q64.64) and about 2^31 units (Q128.128 and wider)
    // of truncation error (before 0.6.4; mpmath gate one_rounding_validation).
    let f = F as i64;
    let l = s.saturating_sub(1) as i64;
    let k = ((f * 10 - 364 + 10 * l + 119) / 120).max(1) as u32;
    s += k - 1;

    // B = A / 2^s
    let mut b = a.copy();
    for _ in 0..s {
        b = b.halve();
    }

    let b2 = b.mat_mul(&b)?;
    let b4 = b2.mat_mul(&b2)?;
    let b6 = b2.mat_mul(&b4)?;
    let id = M::identity(n);
    let c: Vec<M::S> = (0..7).map(|k| M::fraction(PADE_B[k].0, PADE_B[k].1)).collect();

    // V = c0*I + c2*B² + c4*B⁴ + c6*B⁶  (even terms)
    let v = id.scalar_mul(c[0])?
        .add(&b2.scalar_mul(c[2])?)?
        .add(&b4.scalar_mul(c[4])?)?
        .add(&b6.scalar_mul(c[6])?)?;
    // U = B (c1*I + c3*B² + c5*B⁴)
    let p_odd = M::identity(n).scalar_mul(c[1])?
        .add(&b2.scalar_mul(c[3])?)?
        .add(&b4.scalar_mul(c[5])?)?;
    let u = b.mat_mul(&p_odd)?;

    // Solve (V - U) R = V + U, then square s times
    let mut result = M::solve(&v.sub(&u)?, &v.add(&u)?)?;
    for _ in 0..s {
        result = result.mat_mul(&result)?;
    }
    Ok(result)
}

/// Compute-tier internal: exp(A) where A is already a ComputeMatrix.
/// Returns ComputeMatrix, no downscale. Used by matrix_pow for chaining.
/// Which working matrix a format's matrix functions run on: the 32-bit
/// word runs at Q64.64 (`WideMatrix`), every wider word at its compute
/// tier. The routing of 0.6.4, written once per width.
pub trait MatrixFunctions: FixedNum {
    #[doc(hidden)]
    fn exp(a: &FixedMatrix<Self>) -> Result<FixedMatrix<Self>, OverflowDetected>;
    #[doc(hidden)]
    fn sqrt(a: &FixedMatrix<Self>) -> Result<FixedMatrix<Self>, OverflowDetected>;
    #[doc(hidden)]
    fn log(a: &FixedMatrix<Self>) -> Result<FixedMatrix<Self>, OverflowDetected>;
    #[doc(hidden)]
    fn pow(a: &FixedMatrix<Self>, p: Self) -> Result<FixedMatrix<Self>, OverflowDetected>;
}

/// The compute-tier forms of the matrix functions, for the geometry layers.
pub(crate) trait MatrixFunctionsCompute: FixedNum {
    fn exp_compute(a: &ComputeMatrix<Self>) -> Result<ComputeMatrix<Self>, OverflowDetected>;
    fn sqrt_compute(a: &ComputeMatrix<Self>) -> Result<ComputeMatrix<Self>, OverflowDetected>;
    fn log_compute(a: &ComputeMatrix<Self>) -> Result<ComputeMatrix<Self>, OverflowDetected>;
}

impl<const F: u32> MatrixFunctions for Fixed<i32, F> {
    fn exp(a: &FixedMatrix<Self>) -> Result<FixedMatrix<Self>, OverflowDetected> {
        exp_wide::<i32, F, WideMatrix>(&WideMatrix::from_fixed_matrix(a))?.to_fixed_matrix()
    }
    fn sqrt(a: &FixedMatrix<Self>) -> Result<FixedMatrix<Self>, OverflowDetected> {
        sqrt_wide::<i32, F, WideMatrix>(&WideMatrix::from_fixed_matrix(a))?.to_fixed_matrix()
    }
    fn log(a: &FixedMatrix<Self>) -> Result<FixedMatrix<Self>, OverflowDetected> {
        log_wide::<i32, F, WideMatrix>(&WideMatrix::from_fixed_matrix(a))?.to_fixed_matrix()
    }
    fn pow(a: &FixedMatrix<Self>, p: Self) -> Result<FixedMatrix<Self>, OverflowDetected> {
        let log_a = log_wide::<i32, F, WideMatrix>(&WideMatrix::from_fixed_matrix(a))?;
        exp_wide::<i32, F, WideMatrix>(&log_a.scalar_mul(WideMatrix::scalar_from_storage(p))?)?.to_fixed_matrix()
    }
}

impl<const F: u32> MatrixFunctionsCompute for Fixed<i32, F> {
    fn exp_compute(a: &ComputeMatrix<Self>) -> Result<ComputeMatrix<Self>, OverflowDetected> {
        exp_wide::<i32, F, WideMatrix>(&WideMatrix::from_compute(a))?.to_compute()
    }
    fn sqrt_compute(a: &ComputeMatrix<Self>) -> Result<ComputeMatrix<Self>, OverflowDetected> {
        sqrt_wide::<i32, F, WideMatrix>(&WideMatrix::from_compute(a))?.to_compute()
    }
    fn log_compute(a: &ComputeMatrix<Self>) -> Result<ComputeMatrix<Self>, OverflowDetected> {
        log_wide::<i32, F, WideMatrix>(&WideMatrix::from_compute(a))?.to_compute()
    }
}

macro_rules! compute_tier_matrix_functions {
    ($s:ty, $f:expr) => {
        impl MatrixFunctions for Fixed<$s, $f> {
            fn exp(a: &FixedMatrix<Self>) -> Result<FixedMatrix<Self>, OverflowDetected> {
                to_storage(&exp_wide::<$s, $f, _>(&ComputeMatrix::from_fixed_matrix(a))?)
            }
            fn sqrt(a: &FixedMatrix<Self>) -> Result<FixedMatrix<Self>, OverflowDetected> {
                to_storage(&sqrt_wide::<$s, $f, _>(&ComputeMatrix::from_fixed_matrix(a))?)
            }
            fn log(a: &FixedMatrix<Self>) -> Result<FixedMatrix<Self>, OverflowDetected> {
                to_storage(&log_wide::<$s, $f, _>(&ComputeMatrix::from_fixed_matrix(a))?)
            }
            fn pow(a: &FixedMatrix<Self>, p: Self) -> Result<FixedMatrix<Self>, OverflowDetected> {
                let log_a = log_wide::<$s, $f, _>(&ComputeMatrix::from_fixed_matrix(a))?;
                to_storage(&exp_wide::<$s, $f, _>(&log_a.scalar_mul(Self::upscale_to_compute(p.raw())))?)
            }
        }
        impl MatrixFunctionsCompute for Fixed<$s, $f> {
            fn exp_compute(a: &ComputeMatrix<Self>) -> Result<ComputeMatrix<Self>, OverflowDetected> { exp_wide::<$s, $f, _>(a) }
            fn sqrt_compute(a: &ComputeMatrix<Self>) -> Result<ComputeMatrix<Self>, OverflowDetected> { sqrt_wide::<$s, $f, _>(a) }
            fn log_compute(a: &ComputeMatrix<Self>) -> Result<ComputeMatrix<Self>, OverflowDetected> { log_wide::<$s, $f, _>(a) }
        }
    };
}
compute_tier_matrix_functions!(i64, 32);
compute_tier_matrix_functions!(i128, 64);
compute_tier_matrix_functions!(I256, 128);
compute_tier_matrix_functions!(I512, 256);

/// Matrix exponential at the compute tier, for the geometry layers.
pub(crate) fn matrix_exp_compute<T: MatrixFunctionsCompute>(a: &ComputeMatrix<T>) -> Result<ComputeMatrix<T>, OverflowDetected> {
    T::exp_compute(a)
}

/// Matrix exponential via Pade approximation with scaling and squaring,
/// computed on the working matrix of the format and rounded once.
///
/// # Panics
/// Panics if the matrix is not square.
pub fn matrix_exp<S: Storage, const F: u32>(a: &FixedMatrix<Fixed<S, F>>) -> Result<FixedMatrix<Fixed<S, F>>, OverflowDetected>
where
    Fixed<S, F>: MatrixFunctions<Storage = S>,
{
    assert!(a.is_square(), "matrix_exp: matrix must be square");
    <Fixed<S, F> as MatrixFunctions>::exp(a)
}

/// A compute-tier matrix rounded to storage, entry by entry (checked).
fn to_storage<S: Storage, const F: u32>(m: &ComputeMatrix<Fixed<S, F>>) -> Result<FixedMatrix<Fixed<S, F>>, OverflowDetected>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    let mut out = FixedMatrix::new(m.rows(), m.cols());
    for r in 0..m.rows() {
        for c in 0..m.cols() {
            out.set(r, c, Fixed::from_raw(Fixed::<S, F>::downscale_to_storage(m.get(r, c))?));
        }
    }
    Ok(out)
}

fn sqrt_wide<S: Storage, const F: u32, M: FnMatrix<S, F>>(a: &M) -> Result<M, OverflowDetected>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    let n = a.dim();
    let max_iter = 50;
    // Stop once a step is below 2^-(F + 8) relative to ||A||, compared at the
    // working precision: the iteration converges quadratically, so the error
    // left is about 2^-(2F + 16) relative, small enough to survive
    // matrix_log's 2^s unscaling. A stop at one storage unit (2^-F) left
    // 2^-2F relative, 5 units in logm at 8 fraction bits; the older
    // sqrt(quantum) stop (on realtime a fixed 2^-8) left 122 units at 24.
    let bits = F + 8;

    let mut y = a.copy();
    let mut z = M::identity(n);
    for _ in 0..max_iter {
        let y_prev = y.copy();
        let z_inv = z.inverse()?;
        let y_inv = y.inverse()?;
        y = y.add(&z_inv)?.halve();
        z = z.add(&y_inv)?.halve();
        if y.sub(&y_prev)?.step_below(a, bits) {
            return Ok(y);
        }
    }
    Ok(y)
}

/// Compute-tier internal: sqrt(A) where A is already a ComputeMatrix.
/// Returns ComputeMatrix, no downscale. Used by matrix_log_compute for chaining.
/// Matrix square root at the compute tier, for the geometry layers.
pub(crate) fn matrix_sqrt_compute<T: MatrixFunctionsCompute>(a: &ComputeMatrix<T>) -> Result<ComputeMatrix<T>, OverflowDetected> {
    T::sqrt_compute(a)
}

/// Principal matrix square root by the Denman-Beavers iteration on the
/// working matrix of the format, rounded once.
///
/// # Panics
/// Panics if the matrix is not square.
pub fn matrix_sqrt<S: Storage, const F: u32>(a: &FixedMatrix<Fixed<S, F>>) -> Result<FixedMatrix<Fixed<S, F>>, OverflowDetected>
where
    Fixed<S, F>: MatrixFunctions<Storage = S>,
{
    assert!(a.is_square(), "matrix_sqrt: matrix must be square");
    <Fixed<S, F> as MatrixFunctions>::sqrt(a)
}

fn log_wide<S: Storage, const F: u32, M: FnMatrix<S, F>>(a: &M) -> Result<M, OverflowDetected>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    let n = a.dim();
    let id = M::identity(n);
    let quarter = Fixed::<S, F>::one().div_count(4);

    // Phase 1: square roots until ||A_s - I|| < 0.25
    let mut a_s = a.copy();
    let mut s = 0u32;
    for _ in 0..30 {
        if a_s.sub(&id)?.frobenius_below(quarter) {
            break;
        }
        a_s = sqrt_wide::<S, F, M>(&a_s)?;
        s += 1;
    }
    // ... then until ||X|| < 2^-m: the 22-term series leaves ||X||^23 / 23,
    // amplified by the 2^s of the unscaling, which must stay below 2^-(F + 6):
    // 22m >= F + 6 + s. m = 2 (the rule above) at realtime and compact splits;
    // 4 on Q64.64, 7 on Q128.128, 13 on Q256.256, where ||X|| < 0.25 alone
    // left the truncation far above one unit (before 0.6.4).
    let m = ((F + 6 + s + 21) / 22).max(2);
    let mut threshold = quarter;
    for _ in 2..m { threshold = threshold.div_count(2); }
    for _ in 0..30 {
        if a_s.sub(&id)?.frobenius_below(threshold) {
            break;
        }
        a_s = sqrt_wide::<S, F, M>(&a_s)?;
        s += 1;
    }

    // Phase 2: Horner series of log(I + X):
    // X (I - X/2 (I - 2X/3 (I - 3X/4 ...)))
    let x = a_s.sub(&id)?;
    let num_terms = 22;
    let mut horner = M::identity(n);
    for k in (1..num_terms).rev() {
        let x_scaled = x.scalar_mul(M::fraction(k as i64, (k + 1) as i64))?;
        horner = id.sub(&x_scaled.mat_mul(&horner)?)?;
    }
    let mut log_approx = x.mat_mul(&horner)?;

    // Phase 3: log(A) = 2^s log(A_s)
    for _ in 0..s {
        log_approx = log_approx.add(&log_approx)?;
    }
    Ok(log_approx)
}

/// Compute-tier internal: log(A) where A is already a ComputeMatrix.
/// Returns ComputeMatrix, no downscale. Used by matrix_pow for chaining.
/// Matrix logarithm at the compute tier, for the geometry layers.
pub(crate) fn matrix_log_compute<T: MatrixFunctionsCompute>(a: &ComputeMatrix<T>) -> Result<ComputeMatrix<T>, OverflowDetected> {
    T::log_compute(a)
}

/// Principal matrix logarithm by inverse scaling and squaring on the
/// working matrix of the format, rounded once.
///
/// # Panics
/// Panics if the matrix is not square.
pub fn matrix_log<S: Storage, const F: u32>(a: &FixedMatrix<Fixed<S, F>>) -> Result<FixedMatrix<Fixed<S, F>>, OverflowDetected>
where
    Fixed<S, F>: MatrixFunctions<Storage = S>,
{
    assert!(a.is_square(), "matrix_log: matrix must be square");
    <Fixed<S, F> as MatrixFunctions>::log(a)
}

/// `A^p = exp(p log A)`; `p` of 0, 1 and -1 are the identity, `A` and the
/// inverse without the logarithm.
///
/// # Panics
/// Panics if the matrix is not square.
pub fn matrix_pow<S: Storage, const F: u32>(a: &FixedMatrix<Fixed<S, F>>, p: Fixed<S, F>) -> Result<FixedMatrix<Fixed<S, F>>, OverflowDetected>
where
    Fixed<S, F>: MatrixFunctions<Storage = S>,
{
    assert!(a.is_square(), "matrix_pow: matrix must be square");
    let n = a.rows();
    if p.is_zero() {
        return Ok(FixedMatrix::identity(n));
    }
    if p == Fixed::one() {
        return Ok(a.clone());
    }
    if p == -Fixed::<S, F>::one() {
        return lu_decompose(a)?.inverse();
    }
    <Fixed<S, F> as MatrixFunctions>::pow(a, p)
}
