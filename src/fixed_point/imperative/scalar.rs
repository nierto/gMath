//! The bound for code generic over the scalar: every `Fixed<S, F>`.

use super::derived::inverse;
use super::{FixedMatrix, FixedVector};
use crate::fixed_point::core_types::errors::OverflowDetected;
use crate::fixed_point::fixed_num::{Fixed, FixedNum};
use crate::fixed_point::width::Storage;
use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// Every fixed-point format, `Fixed<S, F>` for each admitted storage word
/// and split: the bound for code generic over the scalar, such as the
/// geometry traits (`Manifold<T>`, `OdeSystem<T>`, ...). It carries the
/// operators and comparisons and, hidden, the few operations the traits'
/// default methods need. Sealed: the library implements it for every format
/// and nothing else can.
pub trait Scalar:
    FixedNum
    + Copy
    + PartialOrd
    + Debug
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
    #[doc(hidden)]
    fn sqrt_scalar(self) -> Self;
    #[doc(hidden)]
    fn invert_matrix(m: &FixedMatrix<Self>) -> Result<FixedMatrix<Self>, OverflowDetected>;
    #[doc(hidden)]
    fn sub_vectors(a: &FixedVector<Self>, b: &FixedVector<Self>) -> FixedVector<Self>;
}

impl<S: Storage, const F: u32> Scalar for Fixed<S, F>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    fn sqrt_scalar(self) -> Self { self.sqrt() }
    fn invert_matrix(m: &FixedMatrix<Self>) -> Result<FixedMatrix<Self>, OverflowDetected> { inverse(m) }
    fn sub_vectors(a: &FixedVector<Self>, b: &FixedVector<Self>) -> FixedVector<Self> { a - b }
}
