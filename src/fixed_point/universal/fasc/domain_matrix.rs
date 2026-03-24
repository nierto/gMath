//! Multi-Domain Matrix — StackValue-tagged matrix entries for cross-domain operations.
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
//! - Financial: Decimal interest-rate matrices with 0-ULP exact arithmetic
//! - Neural: Balanced ternary weight matrices for {-1, 0, +1} quantization
//! - Mixed: Decimal input → binary transcendental → decimal output chains
//!
//! **USAGE**:
//! ```rust
//! use g_math::canonical::{gmath, evaluate, DomainMatrix};
//!
//! // Create a 2x2 decimal matrix
//! let m = DomainMatrix::from_values(2, 2, vec![
//!     evaluate(&gmath("0.10")).unwrap(),
//!     evaluate(&gmath("0.20")).unwrap(),
//!     evaluate(&gmath("0.30")).unwrap(),
//!     evaluate(&gmath("0.40")).unwrap(),
//! ]);
//! ```

use crate::fixed_point::imperative::{FixedPoint, FixedMatrix};
use crate::fixed_point::universal::fasc::stack_evaluator::{StackValue, StackEvaluator, evaluate};
use crate::fixed_point::universal::fasc::lazy_expr::LazyExpr;
use crate::fixed_point::universal::ugod::DomainType;
use crate::fixed_point::core_types::errors::OverflowDetected;
use crate::deployment_profiles::DeploymentProfile;

use core::cell::RefCell;

// ============================================================================
// DomainMatrix — StackValue-tagged matrix
// ============================================================================

/// Matrix with domain-tagged elements.
///
/// Each element is a `StackValue` (Binary, Decimal, Ternary, or Symbolic).
/// Operations preserve domain when possible, falling back through rational
/// for cross-domain arithmetic.
#[derive(Debug, Clone)]
pub struct DomainMatrix {
    rows: usize,
    cols: usize,
    data: Vec<StackValue>,
}

impl DomainMatrix {
    /// Create a new DomainMatrix from a flat vector of StackValues (row-major).
    pub fn from_values(rows: usize, cols: usize, data: Vec<StackValue>) -> Self {
        assert_eq!(data.len(), rows * cols, "DomainMatrix: data length mismatch");
        Self { rows, cols, data }
    }

    /// Create a DomainMatrix by evaluating string literals through FASC.
    ///
    /// Each string is parsed and routed to its natural domain:
    /// - "0.10" → Decimal (exact base-10)
    /// - "256" → Binary (exact integer)
    /// - "1/3" → Symbolic (exact rational)
    pub fn from_strings(rows: usize, cols: usize, values: &[&'static str]) -> Result<Self, OverflowDetected> {
        assert_eq!(values.len(), rows * cols, "DomainMatrix: values length mismatch");
        let data: Result<Vec<StackValue>, OverflowDetected> = values.iter()
            .map(|s| evaluate(&LazyExpr::Literal(s)))
            .collect();
        Ok(Self { rows, cols, data: data? })
    }

    /// Create from a FixedMatrix — all entries become Binary domain.
    pub fn from_fixed_matrix(m: &FixedMatrix) -> Self {
        let mut data = Vec::with_capacity(m.rows() * m.cols());
        for r in 0..m.rows() {
            for c in 0..m.cols() {
                data.push(m.get(r, c).to_stack_value());
            }
        }
        Self { rows: m.rows(), cols: m.cols(), data }
    }

    /// Convert to FixedMatrix — all entries materialized to binary storage tier.
    ///
    /// Non-binary entries are converted through the evaluator's binary conversion path.
    pub fn to_fixed_matrix(&self) -> Result<FixedMatrix, OverflowDetected> {
        with_evaluator(|eval| {
            let mut data = Vec::with_capacity(self.rows * self.cols);
            for sv in &self.data {
                match sv.as_binary_storage() {
                    Some(raw) => data.push(FixedPoint::from_raw(raw)),
                    None => {
                        // Convert to binary via evaluator's domain conversion
                        let binary_sv = eval.to_binary_value(sv)?;
                        let materialized = eval.materialize_compute(binary_sv)?;
                        match materialized.as_binary_storage() {
                            Some(raw) => data.push(FixedPoint::from_raw(raw)),
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
    pub fn get(&self, row: usize, col: usize) -> &StackValue {
        &self.data[row * self.cols + col]
    }

    /// Set element at (row, col).
    pub fn set(&mut self, row: usize, col: usize, val: StackValue) {
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

    /// Transpose — domain tags preserved, no computation.
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
    pub fn add(&self, other: &DomainMatrix) -> Result<DomainMatrix, OverflowDetected> {
        assert_eq!(self.rows, other.rows, "DomainMatrix add: row mismatch");
        assert_eq!(self.cols, other.cols, "DomainMatrix add: col mismatch");
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
    pub fn sub(&self, other: &DomainMatrix) -> Result<DomainMatrix, OverflowDetected> {
        assert_eq!(self.rows, other.rows, "DomainMatrix sub: row mismatch");
        assert_eq!(self.cols, other.cols, "DomainMatrix sub: col mismatch");
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
    pub fn mat_mul(&self, other: &DomainMatrix) -> Result<DomainMatrix, OverflowDetected> {
        assert_eq!(self.cols, other.rows, "DomainMatrix matmul: dimension mismatch");
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
    pub fn neg(&self) -> Result<DomainMatrix, OverflowDetected> {
        with_evaluator(|eval| {
            let mut data = Vec::with_capacity(self.data.len());
            for sv in &self.data {
                data.push(eval.negate_value(sv.clone())?);
            }
            Ok(DomainMatrix { rows: self.rows, cols: self.cols, data })
        })
    }

    /// Scalar-matrix multiplication. Scalar is a StackValue (any domain).
    pub fn scalar_mul(&self, s: &StackValue) -> Result<DomainMatrix, OverflowDetected> {
        with_evaluator(|eval| {
            let mut data = Vec::with_capacity(self.data.len());
            for sv in &self.data {
                data.push(eval.multiply_values(s.clone(), sv.clone())?);
            }
            Ok(DomainMatrix { rows: self.rows, cols: self.cols, data })
        })
    }

    /// Trace: sum of diagonal elements (cross-domain accumulation).
    pub fn trace(&self) -> Result<StackValue, OverflowDetected> {
        assert_eq!(self.rows, self.cols, "DomainMatrix trace: not square");
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
// Thread-local evaluator access for DomainMatrix arithmetic
// ============================================================================

thread_local! {
    static DOMAIN_EVALUATOR: RefCell<StackEvaluator> = RefCell::new(
        StackEvaluator::new(compile_time_profile())
    );
}

const fn compile_time_profile() -> DeploymentProfile {
    #[cfg(table_format = "q256_256")]
    { DeploymentProfile::Scientific }
    #[cfg(table_format = "q128_128")]
    { DeploymentProfile::Balanced }
    #[cfg(table_format = "q64_64")]
    { DeploymentProfile::Embedded }
    #[cfg(table_format = "q32_32")]
    { DeploymentProfile::Compact }
    #[cfg(table_format = "q16_16")]
    { DeploymentProfile::Realtime }
}

/// Execute a closure with access to the thread-local evaluator.
fn with_evaluator<T>(f: impl FnOnce(&mut StackEvaluator) -> Result<T, OverflowDetected>) -> Result<T, OverflowDetected> {
    DOMAIN_EVALUATOR.with(|eval| {
        let mut evaluator = eval.borrow_mut();
        evaluator.reset();
        f(&mut evaluator)
    })
}

// ============================================================================
// Display
// ============================================================================

impl core::fmt::Display for DomainMatrix {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "DomainMatrix({}x{}, {:?})", self.rows, self.cols, self.dominant_domain())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_domain_matrix_from_strings_binary() {
        let m = DomainMatrix::from_strings(2, 2, &["1", "2", "3", "4"]).unwrap();
        assert_eq!(m.rows(), 2);
        assert_eq!(m.cols(), 2);
        assert!(m.is_uniform_domain());
    }

    #[test]
    fn test_domain_matrix_from_strings_decimal() {
        let m = DomainMatrix::from_strings(2, 2, &["0.10", "0.20", "0.30", "0.40"]).unwrap();
        assert_eq!(m.rows(), 2);
        assert!(m.is_uniform_domain());
        assert_eq!(m.dominant_domain(), Some(DomainType::Decimal));
    }

    #[test]
    fn test_domain_matrix_transpose() {
        let m = DomainMatrix::from_strings(2, 3, &["1", "2", "3", "4", "5", "6"]).unwrap();
        let mt = m.transpose();
        assert_eq!(mt.rows(), 3);
        assert_eq!(mt.cols(), 2);
    }

    #[test]
    fn test_domain_matrix_identity() {
        let id = DomainMatrix::identity_binary(3);
        assert_eq!(id.rows(), 3);
        assert_eq!(id.cols(), 3);
        assert!(id.is_uniform_domain());
    }

    #[test]
    fn test_domain_matrix_add() {
        let a = DomainMatrix::from_strings(2, 2, &["1", "2", "3", "4"]).unwrap();
        let b = DomainMatrix::from_strings(2, 2, &["10", "20", "30", "40"]).unwrap();
        let c = a.add(&b).unwrap();
        assert_eq!(c.rows(), 2);
        assert_eq!(c.cols(), 2);
    }

    #[test]
    fn test_domain_matrix_decimal_matmul() {
        // Decimal 2x2 identity * decimal vector-like
        let id = DomainMatrix::from_strings(2, 2, &["1.00", "0.00", "0.00", "1.00"]).unwrap();
        let v = DomainMatrix::from_strings(2, 1, &["0.10", "0.20"]).unwrap();
        let result = id.mat_mul(&v).unwrap();
        assert_eq!(result.rows(), 2);
        assert_eq!(result.cols(), 1);
    }
}
