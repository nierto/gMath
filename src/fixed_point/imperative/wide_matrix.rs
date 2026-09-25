//! Q64.64 matrices for the realtime profile's matrix functions.
//!
//! On realtime the compute tier holds `2F` fractional bits in an i64 (16 at
//! `GMATH_FRAC_BITS = 8`), and scaling and squaring amplifies that resolution:
//! `expm` of a norm-7 matrix was 8 units off at 8 fraction bits. The matrix
//! functions therefore run on i128 values with 64 fractional bits there
//! (40 to 56 guard bits over the compute tier across the gated splits), the
//! width compact's compute tier already has. Inputs widen exactly; results are
//! rounded once, to the compute tier or to storage.
//!
//! Every sum of products is exact on the I256 accumulator (128 fractional
//! bits) and rounded once, to nearest with ties toward +infinity; a quotient
//! of an exact sum by an entry is one rounding of the exact quotient. Every
//! narrowing is checked: leaving the range is a `TierOverflow`, never a wrap.

use super::linalg::ComputeStorage;
use super::compute_matrix::ComputeMatrix;
use super::{FixedMatrix, FixedPoint};
use crate::fixed_point::core_types::errors::OverflowDetected;
use crate::fixed_point::domains::binary_fixed::i256::{divmod_i256_by_i256, mul_i128_to_i256};
use crate::fixed_point::frac_config::FRAC_BITS;
use crate::fixed_point::I256;

const OVERFLOW: OverflowDetected = OverflowDetected::TierOverflow;

/// Fractional bits of a wide value.
const WIDE_FRAC: u32 = 64;

/// A square or rectangular matrix of Q64.64 values (row-major).
#[derive(Clone, Debug)]
pub(crate) struct WideMatrix {
    rows: usize,
    cols: usize,
    data: Vec<i128>,
}

/// `v >> shift` rounded to nearest, ties toward +infinity (`shift >= 1`).
#[inline]
fn round_shift(v: i128, shift: u32) -> i128 {
    (v >> shift) + ((v >> (shift - 1)) & 1)
}

/// An exact I256 value at 128 fractional bits rounded to Q64.64.
#[inline]
fn narrow_128(v: I256) -> Result<i128, OverflowDetected> {
    let half = I256::from_i128(1) << 63usize;
    let q = v.checked_add(half).ok_or(OVERFLOW)? >> 64u32;
    if q.fits_in_i128() { Ok(q.as_i128()) } else { Err(OVERFLOW) }
}

/// `num / den` rounded to nearest (ties toward +infinity) to Q64.64: `num` an
/// exact value at 128 fractional bits, `den` a nonzero Q64.64 value.
fn divide_128(num: I256, den: i128) -> Result<i128, OverflowDetected> {
    if den == 0 {
        return Err(OverflowDetected::DivisionByZero);
    }
    let d = I256::from_i128(den);
    let (q, r) = divmod_i256_by_i256(num, d);
    // truncated quotient; one step away from zero past the half (at the half
    // only for a positive quotient)
    let positive = num.is_negative() == d.is_negative();
    let r_abs = if r.is_negative() { I256::zero() - r } else { r };
    let d_abs = if d.is_negative() { I256::zero() - d } else { d };
    let twice = r_abs.checked_add(r_abs).ok_or(OVERFLOW)?;
    let bump = if positive { twice >= d_abs } else { twice > d_abs };
    let q = if bump {
        if positive { q + I256::from_i128(1) } else { q - I256::from_i128(1) }
    } else {
        q
    };
    if q.fits_in_i128() { Ok(q.as_i128()) } else { Err(OVERFLOW) }
}

/// A Q64.64 value at 128 fractional bits (exact).
#[inline]
fn widen_128(v: i128) -> I256 {
    I256::from_i128(v) << 64usize
}

/// `init - sum a_k b_k`, exact at 128 fractional bits.
fn sub_dot_128(init: i128, a: &[i128], b: &[i128]) -> Result<I256, OverflowDetected> {
    let mut acc = widen_128(init);
    for (x, y) in a.iter().zip(b) {
        acc = acc.checked_sub(mul_i128_to_i256(*x, *y)).ok_or(OVERFLOW)?;
    }
    Ok(acc)
}

impl WideMatrix {
    pub(crate) fn new(rows: usize, cols: usize) -> Self {
        WideMatrix { rows, cols, data: vec![0; rows * cols] }
    }

    pub(crate) fn identity(n: usize) -> Self {
        let mut m = Self::new(n, n);
        for i in 0..n {
            m.set(i, i, 1i128 << WIDE_FRAC);
        }
        m
    }

    #[inline]
    pub(crate) fn rows(&self) -> usize { self.rows }

    #[inline]
    pub(crate) fn get(&self, r: usize, c: usize) -> i128 { self.data[r * self.cols + c] }

    #[inline]
    pub(crate) fn set(&mut self, r: usize, c: usize, v: i128) { self.data[r * self.cols + c] = v; }

    /// A storage matrix widened exactly.
    pub(crate) fn from_fixed_matrix(m: &FixedMatrix) -> Self {
        let mut w = Self::new(m.rows(), m.cols());
        for r in 0..m.rows() {
            for c in 0..m.cols() {
                w.set(r, c, (m.get(r, c).raw() as i128) << (WIDE_FRAC - FRAC_BITS));
            }
        }
        w
    }

    /// A compute-tier matrix (`2F` fractional bits) widened exactly.
    pub(crate) fn from_compute(m: &ComputeMatrix) -> Self {
        let mut w = Self::new(m.rows(), m.cols());
        for r in 0..m.rows() {
            for c in 0..m.cols() {
                w.set(r, c, (m.get(r, c) as i128) << (WIDE_FRAC - 2 * FRAC_BITS));
            }
        }
        w
    }

    /// Rounded once to storage (checked).
    pub(crate) fn to_fixed_matrix(&self) -> Result<FixedMatrix, OverflowDetected> {
        let mut out = FixedMatrix::new(self.rows, self.cols);
        for r in 0..self.rows {
            for c in 0..self.cols {
                let v = round_shift(self.get(r, c), WIDE_FRAC - FRAC_BITS);
                out.set(r, c, FixedPoint::from_raw(i32::try_from(v).map_err(|_| OVERFLOW)?));
            }
        }
        Ok(out)
    }

    /// Rounded once to the compute tier (checked).
    pub(crate) fn to_compute(&self) -> Result<ComputeMatrix, OverflowDetected> {
        let mut out = ComputeMatrix::new(self.rows, self.cols);
        for r in 0..self.rows {
            for c in 0..self.cols {
                let v = round_shift(self.get(r, c), WIDE_FRAC - 2 * FRAC_BITS);
                out.set(r, c, ComputeStorage::try_from(v).map_err(|_| OVERFLOW)?);
            }
        }
        Ok(out)
    }

    /// A storage scalar as a wide value (exact).
    pub(crate) fn scalar_from_storage(x: FixedPoint) -> i128 {
        (x.raw() as i128) << (WIDE_FRAC - FRAC_BITS)
    }

    /// `num / den` as a wide value, rounded once (`den > 0`).
    pub(crate) fn fraction(num: i64, den: i64) -> i128 {
        divide_128(widen_128(num as i128) << 64usize, (den as i128) << WIDE_FRAC)
            .expect("fraction fits the wide tier")
    }

    fn zip(&self, o: &Self, f: impl Fn(i128, i128) -> Option<i128>) -> Result<Self, OverflowDetected> {
        assert_eq!((self.rows, self.cols), (o.rows, o.cols), "WideMatrix: dimension mismatch");
        let data: Option<Vec<i128>> = self.data.iter().zip(&o.data).map(|(a, b)| f(*a, *b)).collect();
        Ok(WideMatrix { rows: self.rows, cols: self.cols, data: data.ok_or(OVERFLOW)? })
    }

    pub(crate) fn add(&self, o: &Self) -> Result<Self, OverflowDetected> { self.zip(o, i128::checked_add) }

    pub(crate) fn sub(&self, o: &Self) -> Result<Self, OverflowDetected> { self.zip(o, i128::checked_sub) }

    /// Halved to nearest (ties toward +infinity).
    pub(crate) fn halve(&self) -> Self {
        WideMatrix { rows: self.rows, cols: self.cols, data: self.data.iter().map(|v| round_shift(*v, 1)).collect() }
    }

    /// Every entry times `s`, rounded once.
    pub(crate) fn scalar_mul(&self, s: i128) -> Result<Self, OverflowDetected> {
        let data: Result<Vec<i128>, OverflowDetected> =
            self.data.iter().map(|v| narrow_128(mul_i128_to_i256(*v, s))).collect();
        Ok(WideMatrix { rows: self.rows, cols: self.cols, data: data? })
    }

    /// Matrix product, each entry one rounding of its exact sum.
    pub(crate) fn mat_mul(&self, o: &Self) -> Result<Self, OverflowDetected> {
        assert_eq!(self.cols, o.rows, "WideMatrix::mat_mul: dimension mismatch");
        let mut out = Self::new(self.rows, o.cols);
        for r in 0..self.rows {
            for c in 0..o.cols {
                let a: Vec<i128> = (0..self.cols).map(|k| -self.get(r, k)).collect();
                let b: Vec<i128> = (0..self.cols).map(|k| o.get(k, c)).collect();
                out.set(r, c, narrow_128(sub_dot_128(0, &a, &b)?)?);
            }
        }
        Ok(out)
    }

    /// True when every entry is exactly zero.
    pub(crate) fn is_zero(&self) -> bool { self.data.iter().all(|v| *v == 0) }

    /// `||M||_1` rounded to storage (for step counts only; saturates at the
    /// storage maximum, which only ever means "large").
    pub(crate) fn norm_1(&self) -> FixedPoint {
        let mut best: i128 = 0;
        for c in 0..self.cols {
            let mut sum: i128 = 0;
            for r in 0..self.rows {
                sum = sum.saturating_add(self.get(r, c).saturating_abs());
            }
            best = best.max(sum);
        }
        let v = round_shift(best, WIDE_FRAC - FRAC_BITS);
        FixedPoint::from_raw(v.min(i32::MAX as i128) as i32)
    }

    /// `||M||_F` below the storage value `t`: an exact comparison of the sum
    /// of squares with `t^2`.
    pub(crate) fn frobenius_below(&self, t: FixedPoint) -> bool {
        let mut sum = I256::zero();
        for v in &self.data {
            match sum.checked_add(mul_i128_to_i256(*v, *v)) {
                Some(s) => sum = s,
                None => return false,
            }
        }
        let tw = Self::scalar_from_storage(t);
        sum < mul_i128_to_i256(tw, tw)
    }

    /// `||self||_F < ||reference||_F * 2^-bits`, from exact sums of squares.
    pub(crate) fn step_below(&self, reference: &Self, bits: u32) -> bool {
        let sumsq = |m: &WideMatrix| -> Option<I256> {
            let mut sum = I256::zero();
            for v in &m.data {
                sum = sum.checked_add(mul_i128_to_i256(*v, *v))?;
            }
            Some(sum)
        };
        match (sumsq(self), sumsq(reference)) {
            (Some(step), Some(r)) => step < (r >> (2 * bits)),
            _ => false,
        }
    }

    /// LU with partial pivoting (the pivot rule of the compute-tier LU), every
    /// entry one rounding of an exact sum or quotient.
    pub(crate) fn lu(&self) -> Result<WideLU, OverflowDetected> {
        let n = self.rows;
        let mut pa = self.clone();
        let mut l = Self::new(n, n);
        let mut u = Self::new(n, n);
        let mut perm: Vec<usize> = (0..n).collect();
        for k in 0..n {
            let mut max_abs = I256::zero();
            let mut max_row = k;
            for i in k..n {
                let l_row: Vec<i128> = (0..k).map(|m| l.get(i, m)).collect();
                let u_col: Vec<i128> = (0..k).map(|m| u.get(m, k)).collect();
                let cand = sub_dot_128(pa.get(i, k), &l_row, &u_col)?;
                let abs = if cand.is_negative() { I256::zero() - cand } else { cand };
                if abs > max_abs {
                    max_abs = abs;
                    max_row = i;
                }
            }
            if max_abs.is_zero() {
                return Err(OverflowDetected::DivisionByZero);
            }
            if max_row != k {
                for c in 0..n {
                    let t = pa.get(k, c);
                    pa.set(k, c, pa.get(max_row, c));
                    pa.set(max_row, c, t);
                }
                perm.swap(k, max_row);
                for j in 0..k {
                    let t = l.get(k, j);
                    l.set(k, j, l.get(max_row, j));
                    l.set(max_row, j, t);
                }
            }
            for j in k..n {
                let l_row: Vec<i128> = (0..k).map(|m| l.get(k, m)).collect();
                let u_col: Vec<i128> = (0..k).map(|m| u.get(m, j)).collect();
                u.set(k, j, narrow_128(sub_dot_128(pa.get(k, j), &l_row, &u_col)?)?);
            }
            l.set(k, k, 1i128 << WIDE_FRAC);
            let pivot = u.get(k, k);
            for i in (k + 1)..n {
                let l_row: Vec<i128> = (0..k).map(|m| l.get(i, m)).collect();
                let u_col: Vec<i128> = (0..k).map(|m| u.get(m, k)).collect();
                l.set(i, k, divide_128(sub_dot_128(pa.get(i, k), &l_row, &u_col)?, pivot)?);
            }
        }
        Ok(WideLU { l, u, perm })
    }
}

/// `PA = LU` of a wide matrix.
pub(crate) struct WideLU {
    l: WideMatrix,
    u: WideMatrix,
    perm: Vec<usize>,
}

impl WideLU {
    /// `A x = b`, each entry one rounding of an exact sum (forward) or of an
    /// exact quotient (back substitution).
    pub(crate) fn solve(&self, b: &[i128]) -> Result<Vec<i128>, OverflowDetected> {
        let n = self.l.rows();
        let pb: Vec<i128> = (0..n).map(|i| b[self.perm[i]]).collect();
        let mut y = vec![0i128; n];
        for i in 0..n {
            let l_row: Vec<i128> = (0..i).map(|j| self.l.get(i, j)).collect();
            y[i] = narrow_128(sub_dot_128(pb[i], &l_row, &y[..i])?)?;
        }
        let mut x = vec![0i128; n];
        for i in (0..n).rev() {
            let u_row: Vec<i128> = ((i + 1)..n).map(|j| self.u.get(i, j)).collect();
            x[i] = divide_128(sub_dot_128(y[i], &u_row, &x[i + 1..])?, self.u.get(i, i))?;
        }
        Ok(x)
    }

    /// `A^-1`, column by column.
    pub(crate) fn inverse(&self) -> Result<WideMatrix, OverflowDetected> {
        let n = self.l.rows();
        let mut inv = WideMatrix::new(n, n);
        for j in 0..n {
            let e: Vec<i128> = (0..n).map(|i| if i == j { 1i128 << WIDE_FRAC } else { 0 }).collect();
            let col = self.solve(&e)?;
            for i in 0..n {
                inv.set(i, j, col[i]);
            }
        }
        Ok(inv)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Conversions are exact up and round once down; products and quotients
    /// round once from their exact values.
    #[test]
    fn wide_arithmetic_rounds_once() {
        let third = WideMatrix::fraction(1, 3);
        // 2^64 / 3 = 6148914691236517205.33..: nearest is ...205
        assert_eq!(third, 6148914691236517205);
        assert_eq!(WideMatrix::fraction(2, 3), 12297829382473034411);
        let mut m = WideMatrix::identity(2);
        m.set(0, 1, third);
        let sq = m.mat_mul(&m).unwrap();
        // [[1, 1/3], [0, 1]]^2 = [[1, 2/3], [0, 1]]: 2 * third exactly
        assert_eq!(sq.get(0, 1), 2 * third);
        let inv = m.lu().unwrap().inverse().unwrap();
        assert_eq!(inv.get(0, 1), -third);
        let f = FixedMatrix::identity(2);
        assert_eq!(WideMatrix::from_fixed_matrix(&f).to_fixed_matrix().unwrap(), f);
    }
}
