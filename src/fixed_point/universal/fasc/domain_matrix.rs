//! Multi-Domain Matrix: StackValue-tagged matrix entries for cross-domain operations.
//!
//! **MISSION**: Enable matrix operations across all four domains (binary, decimal,
//! ternary, symbolic) with automatic cross-domain routing via the FASC evaluator.
//!
//! **ARCHITECTURE**: Each element is a `StackValue` carrying its own domain tag.
//! Same-domain operations use native dispatch; cross-domain operations convert
//! through rational representation. Transcendentals route through binary compute
//! tier (the only domain with transcendental tables).
//!
//! **USE CASES**:
//! - Financial: Decimal interest-rate matrices with exact decimal arithmetic
//! - Neural: Balanced ternary weight matrices for {-1, 0, +1} quantization
//! - Mixed: Decimal input → binary transcendental → decimal output chains
//!
//! **USAGE**:
//! ```rust
//! use g_math::canonical::{gmath, evaluate, DomainMatrix};
//! use g_math::fixed_point::Q64_64;
//!
//! // Create a 2x2 decimal matrix
//! let m = DomainMatrix::from_values(2, 2, vec![
//!     evaluate::<Q64_64>(&gmath("0.10")).unwrap(),
//!     evaluate(&gmath("0.20")).unwrap(),
//!     evaluate(&gmath("0.30")).unwrap(),
//!     evaluate(&gmath("0.40")).unwrap(),
//! ]);
//! ```

use crate::fixed_point::imperative::FixedMatrix;
use crate::fixed_point::universal::fasc::stack_evaluator::{evaluate, with_evaluator as registry_evaluator, FascFormat, StackEvaluator, StackValue};
use crate::fixed_point::fixed_num::{Fixed, FixedNum};
use crate::fixed_point::width::Storage;
use crate::fixed_point::universal::fasc::lazy_expr::LazyExpr;
use crate::fixed_point::universal::ugod::DomainType;
use crate::fixed_point::core_types::errors::OverflowDetected;


// ============================================================================
// DomainMatrix — StackValue-tagged matrix
// ============================================================================

/// Matrix with domain-tagged elements.
///
/// Each element is a `StackValue` (Binary, Decimal, Ternary, or Symbolic).
/// Operations preserve domain when possible, falling back through rational
/// for cross-domain arithmetic.
#[derive(Debug, Clone)]
pub struct DomainMatrix<T: FascFormat> {
    rows: usize,
    cols: usize,
    data: Vec<StackValue<T>>,
}

impl<S: Storage, const F: u32> DomainMatrix<Fixed<S, F>>
where
    Fixed<S, F>: FascFormat + FixedNum<Storage = S>,
{
    /// Create a new DomainMatrix<Fixed<S, F>> from a flat vector of StackValues (row-major).
    pub fn from_values(rows: usize, cols: usize, data: Vec<StackValue<Fixed<S, F>>>) -> Self {
        assert_eq!(data.len(), rows * cols, "DomainMatrix<Fixed<S, F>>: data length mismatch");
        Self { rows, cols, data }
    }

    /// Create a DomainMatrix<Fixed<S, F>> by evaluating string literals through FASC.
    ///
    /// Each string is parsed and routed to its natural domain:
    /// - "0.10" → Decimal (exact base-10)
    /// - "256" → Binary (exact integer)
    /// - "1/3" → Symbolic (exact rational)
    pub fn from_strings(rows: usize, cols: usize, values: &[&'static str]) -> Result<Self, OverflowDetected> {
        assert_eq!(values.len(), rows * cols, "DomainMatrix<Fixed<S, F>>: values length mismatch");
        let data: Result<Vec<StackValue<Fixed<S, F>>>, OverflowDetected> = values.iter()
            .map(|s| evaluate(&LazyExpr::<Fixed<S, F>>::Literal(s)))
            .collect();
        Ok(Self { rows, cols, data: data? })
    }

    /// Create from a FixedMatrix<Fixed<S, F>>: all entries become Binary domain.
    pub fn from_fixed_matrix(m: &FixedMatrix<Fixed<S, F>>) -> Self {
        let mut data = Vec::with_capacity(m.rows() * m.cols());
        for r in 0..m.rows() {
            for c in 0..m.cols() {
                data.push(m.get(r, c).to_stack_value());
            }
        }
        Self { rows: m.rows(), cols: m.cols(), data }
    }

    /// Convert to FixedMatrix<Fixed<S, F>>: all entries materialized to binary storage tier.
    ///
    /// Non-binary entries are converted through the evaluator's binary conversion path.
    pub fn to_fixed_matrix(&self) -> Result<FixedMatrix<Fixed<S, F>>, OverflowDetected> {
        with_evaluator(|eval| {
            let mut data = Vec::with_capacity(self.rows * self.cols);
            for sv in &self.data {
                match sv.as_binary_storage() {
                    Some(raw) => data.push(Fixed::<S, F>::from_raw(raw)),
                    None => {
                        // Convert to binary via evaluator's domain conversion
                        let binary_sv = eval.to_compute_value(sv)?;
                        let materialized = eval.materialize_compute(binary_sv)?;
                        match materialized.as_binary_storage() {
                            Some(raw) => data.push(Fixed::<S, F>::from_raw(raw)),
                            None => return Err(OverflowDetected::InvalidInput),
                        }
                    }
                }
            }
            Ok(FixedMatrix::from_fn(self.rows, self.cols, |r, c| data[r * self.cols + c]))
        })
    }

    /// Number of rows.
    pub fn rows(&self) -> usize { self.rows }

    /// Number of columns.
    pub fn cols(&self) -> usize { self.cols }

    /// Get element at (row, col).
    pub fn get(&self, row: usize, col: usize) -> &StackValue<Fixed<S, F>> {
        &self.data[row * self.cols + col]
    }

    /// Set element at (row, col).
    pub fn set(&mut self, row: usize, col: usize, val: StackValue<Fixed<S, F>>) {
        self.data[row * self.cols + col] = val;
    }

    /// Check if all elements are in the same domain.
    pub fn is_uniform_domain(&self) -> bool {
        if self.data.is_empty() { return true; }
        let first = self.data[0].domain_type();
        self.data.iter().all(|sv| sv.domain_type() == first)
    }

    /// Get the dominant domain (most common among elements).
    pub fn dominant_domain(&self) -> Option<DomainType> {
        if self.data.is_empty() { return None; }
        let mut counts = [0u32; 4]; // Binary, Decimal, Ternary, Symbolic
        for sv in &self.data {
            match sv.domain_type() {
                Some(DomainType::Binary) => counts[0] += 1,
                Some(DomainType::Decimal) => counts[1] += 1,
                Some(DomainType::Ternary) => counts[2] += 1,
                Some(DomainType::Symbolic) => counts[3] += 1,
                _ => {}
            }
        }
        let max_idx = counts.iter().enumerate().max_by_key(|(_, &c)| c).map(|(i, _)| i)?;
        match max_idx {
            0 => Some(DomainType::Binary),
            1 => Some(DomainType::Decimal),
            2 => Some(DomainType::Ternary),
            3 => Some(DomainType::Symbolic),
            _ => None,
        }
    }

    /// Identity matrix in binary domain.
    pub fn identity_binary(n: usize) -> Self {
        Self::from_fixed_matrix(&FixedMatrix::identity(n))
    }

    /// Transpose: domain tags preserved, no computation.
    pub fn transpose(&self) -> Self {
        let mut data = Vec::with_capacity(self.rows * self.cols);
        for c in 0..self.cols {
            for r in 0..self.rows {
                data.push(self.data[r * self.cols + c].clone());
            }
        }
        Self { rows: self.cols, cols: self.rows, data }
    }

    // ========================================================================
    // Arithmetic via FASC evaluator dispatch
    // ========================================================================

    /// Element-wise addition. Cross-domain pairs route through rational.
    pub fn add(&self, other: &DomainMatrix<Fixed<S, F>>) -> Result<DomainMatrix<Fixed<S, F>>, OverflowDetected> {
        assert_eq!(self.rows, other.rows, "DomainMatrix<Fixed<S, F>> add: row mismatch");
        assert_eq!(self.cols, other.cols, "DomainMatrix<Fixed<S, F>> add: col mismatch");
        with_evaluator(|eval| {
            let mut data = Vec::with_capacity(self.rows * self.cols);
            for i in 0..self.data.len() {
                let result = eval.add_values(self.data[i].clone(), other.data[i].clone())?;
                data.push(result);
            }
            Ok(DomainMatrix { rows: self.rows, cols: self.cols, data })
        })
    }

    /// Element-wise subtraction. Cross-domain pairs route through rational.
    pub fn sub(&self, other: &DomainMatrix<Fixed<S, F>>) -> Result<DomainMatrix<Fixed<S, F>>, OverflowDetected> {
        assert_eq!(self.rows, other.rows, "DomainMatrix<Fixed<S, F>> sub: row mismatch");
        assert_eq!(self.cols, other.cols, "DomainMatrix<Fixed<S, F>> sub: col mismatch");
        with_evaluator(|eval| {
            let mut data = Vec::with_capacity(self.rows * self.cols);
            for i in 0..self.data.len() {
                let result = eval.subtract_values(self.data[i].clone(), other.data[i].clone())?;
                data.push(result);
            }
            Ok(DomainMatrix { rows: self.rows, cols: self.cols, data })
        })
    }

    /// Matrix multiplication. Each output element is a dot product
    /// computed via the FASC evaluator's domain-aware arithmetic.
    pub fn mat_mul(&self, other: &DomainMatrix<Fixed<S, F>>) -> Result<DomainMatrix<Fixed<S, F>>, OverflowDetected> {
        assert_eq!(self.cols, other.rows, "DomainMatrix<Fixed<S, F>> matmul: dimension mismatch");
        let k = self.cols;
        with_evaluator(|eval| {
            let mut data = Vec::with_capacity(self.rows * other.cols);
            for r in 0..self.rows {
                for c in 0..other.cols {
                    // Dot product: sum(self[r,m] * other[m,c]) for m in 0..k
                    let first_prod = eval.multiply_values(
                        self.data[r * k].clone(),
                        other.data[c].clone(),
                    )?;
                    let mut acc = first_prod;
                    for m in 1..k {
                        let prod = eval.multiply_values(
                            self.data[r * k + m].clone(),
                            other.data[m * other.cols + c].clone(),
                        )?;
                        acc = eval.add_values(acc, prod)?;
                    }
                    data.push(acc);
                }
            }
            Ok(DomainMatrix { rows: self.rows, cols: other.cols, data })
        })
    }

    /// Element-wise negation. Preserves domain.
    pub fn neg(&self) -> Result<DomainMatrix<Fixed<S, F>>, OverflowDetected> {
        with_evaluator(|eval| {
            let mut data = Vec::with_capacity(self.data.len());
            for sv in &self.data {
                data.push(eval.negate_value(sv.clone())?);
            }
            Ok(DomainMatrix { rows: self.rows, cols: self.cols, data })
        })
    }

    /// Scalar-matrix multiplication. Scalar is a StackValue<Fixed<S, F>> (any domain).
    pub fn scalar_mul(&self, s: &StackValue<Fixed<S, F>>) -> Result<DomainMatrix<Fixed<S, F>>, OverflowDetected> {
        with_evaluator(|eval| {
            let mut data = Vec::with_capacity(self.data.len());
            for sv in &self.data {
                data.push(eval.multiply_values(s.clone(), sv.clone())?);
            }
            Ok(DomainMatrix { rows: self.rows, cols: self.cols, data })
        })
    }

    /// Trace: sum of diagonal elements (cross-domain accumulation).
    pub fn trace(&self) -> Result<StackValue<Fixed<S, F>>, OverflowDetected> {
        assert_eq!(self.rows, self.cols, "DomainMatrix<Fixed<S, F>> trace: not square");
        with_evaluator(|eval| {
            let mut acc = self.data[0].clone();
            for i in 1..self.rows {
                acc = eval.add_values(acc, self.data[i * self.cols + i].clone())?;
            }
            Ok(acc)
        })
    }
}

// ============================================================================
// Thread-local evaluator access for DomainMatrix<Fixed<S, F>> arithmetic
// ============================================================================

/// This thread's evaluator for `T`, reset for a fresh computation.
fn with_evaluator<T: FascFormat, R>(f: impl FnOnce(&mut StackEvaluator<T>) -> Result<R, OverflowDetected>) -> Result<R, OverflowDetected> {
    registry_evaluator(|evaluator: &mut StackEvaluator<T>| {
        evaluator.reset();
        f(evaluator)
    })
}

impl<S: Storage, const F: u32> core::fmt::Display for DomainMatrix<Fixed<S, F>>
where
    Fixed<S, F>: FascFormat + FixedNum<Storage = S>,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "DomainMatrix({}x{}, {:?})", self.rows, self.cols, self.dominant_domain())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    macro_rules! suite {
        ($m:ident, $S:ty, $F:literal, $W:ident) => {
            mod $m {
                #![allow(dead_code, unused_imports, unreachable_code)]
                use super::super::*;
                type FixedPoint = crate::fixed_point::Fixed<$S, $F>;
                type BinaryStorage = $S;
                type ComputeStorage = <$S as crate::fixed_point::Storage>::Compute;
                const FRAC_BITS: u32 = $F;
                const BITS: u32 = <$S as crate::fixed_point::WideInt>::BITS;
                use super::super::*;
                #[allow(unused_imports)]

                #[test]
                fn test_domain_matrix_from_strings_binary() {
                    let m = DomainMatrix::<FixedPoint>::from_strings(2, 2, &["1", "2", "3", "4"]).unwrap();
                    assert_eq!(m.rows(), 2);
                    assert_eq!(m.cols(), 2);
                    assert!(m.is_uniform_domain());
                }

                #[test]
                fn test_domain_matrix_from_strings_decimal() {
                    let m = DomainMatrix::<FixedPoint>::from_strings(2, 2, &["0.10", "0.20", "0.30", "0.40"]).unwrap();
                    assert_eq!(m.rows(), 2);
                    assert!(m.is_uniform_domain());
                    assert_eq!(m.dominant_domain(), Some(DomainType::Decimal));
                }

                #[test]
                fn test_domain_matrix_transpose() {
                    let m = DomainMatrix::<FixedPoint>::from_strings(2, 3, &["1", "2", "3", "4", "5", "6"]).unwrap();
                    let mt = m.transpose();
                    assert_eq!(mt.rows(), 3);
                    assert_eq!(mt.cols(), 2);
                }

                #[test]
                fn test_domain_matrix_identity() {
                    let id = DomainMatrix::<FixedPoint>::identity_binary(3);
                    assert_eq!(id.rows(), 3);
                    assert_eq!(id.cols(), 3);
                    assert!(id.is_uniform_domain());
                }

                #[test]
                fn test_domain_matrix_add() {
                    let a = DomainMatrix::<FixedPoint>::from_strings(2, 2, &["1", "2", "3", "4"]).unwrap();
                    let b = DomainMatrix::<FixedPoint>::from_strings(2, 2, &["10", "20", "30", "40"]).unwrap();
                    let c = a.add(&b).unwrap();
                    assert_eq!(c.rows(), 2);
                    assert_eq!(c.cols(), 2);
                }

                #[test]
                fn test_domain_matrix_decimal_matmul() {
                    // Decimal 2x2 identity * decimal vector-like
                    let id = DomainMatrix::<FixedPoint>::from_strings(2, 2, &["1.00", "0.00", "0.00", "1.00"]).unwrap();
                    let v = DomainMatrix::<FixedPoint>::from_strings(2, 1, &["0.10", "0.20"]).unwrap();
                    let result = id.mat_mul(&v).unwrap();
                    assert_eq!(result.rows(), 2);
                    assert_eq!(result.cols(), 1);
                }
            }
        };
    }
    for_each_format!(suite);
}
