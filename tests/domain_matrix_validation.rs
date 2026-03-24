//! Validation tests for DomainMatrix — multi-domain matrix operations.
//!
//! Tests:
//! 1. Pure binary matrices (baseline)
//! 2. Pure decimal matrices (financial-grade 0-ULP)
//! 3. Mixed-domain operations (cross-domain via rational)
//! 4. Symbolic matrices (exact rational arithmetic)

use g_math::canonical::{gmath, evaluate, DomainMatrix, StackValue};
use g_math::fixed_point::{FixedPoint, FixedMatrix};
use g_math::fixed_point::universal::ugod::DomainType;

fn fp(s: &str) -> FixedPoint {
    if s.starts_with('-') { -FixedPoint::from_str(&s[1..]) }
    else { FixedPoint::from_str(s) }
}

fn sv_to_fp(sv: &StackValue) -> FixedPoint {
    match sv.as_binary_storage() {
        Some(raw) => FixedPoint::from_raw(raw),
        None => {
            // Non-binary domain — convert via formatted string
            let s = format!("{}", sv);
            fp(&s)
        }
    }
}

fn tight() -> FixedPoint { fp("0.000000001") }

fn assert_fp(got: FixedPoint, exp: FixedPoint, tol: FixedPoint, name: &str) {
    let d = (got - exp).abs();
    assert!(d < tol, "{}: got {}, expected {}, diff={}", name, got, exp, d);
}

// ============================================================================
// Binary domain (baseline)
// ============================================================================

#[test]
fn test_binary_domain_matrix_identity() {
    let id = DomainMatrix::identity_binary(3);
    assert_eq!(id.rows(), 3);
    assert_eq!(id.cols(), 3);
    assert!(id.is_uniform_domain());
    assert_eq!(id.dominant_domain(), Some(DomainType::Binary));
}

#[test]
fn test_binary_domain_matrix_from_strings() {
    let m = DomainMatrix::from_strings(2, 2, &["1", "2", "3", "4"]).unwrap();
    assert_eq!(m.rows(), 2);
    assert_eq!(m.cols(), 2);
}

#[test]
fn test_binary_domain_matrix_add() {
    let a = DomainMatrix::from_strings(2, 2, &["1", "2", "3", "4"]).unwrap();
    let b = DomainMatrix::from_strings(2, 2, &["10", "20", "30", "40"]).unwrap();
    let c = a.add(&b).unwrap();
    // [1+10, 2+20; 3+30, 4+40] = [11, 22; 33, 44]
    assert_fp(sv_to_fp(c.get(0, 0)), fp("11"), tight(), "binary_add[0,0]");
    assert_fp(sv_to_fp(c.get(0, 1)), fp("22"), tight(), "binary_add[0,1]");
    assert_fp(sv_to_fp(c.get(1, 0)), fp("33"), tight(), "binary_add[1,0]");
    assert_fp(sv_to_fp(c.get(1, 1)), fp("44"), tight(), "binary_add[1,1]");
}

#[test]
fn test_binary_domain_matrix_matmul() {
    // [[1, 2], [3, 4]] * [[5, 6], [7, 8]] = [[19, 22], [43, 50]]
    let a = DomainMatrix::from_strings(2, 2, &["1", "2", "3", "4"]).unwrap();
    let b = DomainMatrix::from_strings(2, 2, &["5", "6", "7", "8"]).unwrap();
    let c = a.mat_mul(&b).unwrap();
    assert_fp(sv_to_fp(c.get(0, 0)), fp("19"), tight(), "matmul[0,0]");
    assert_fp(sv_to_fp(c.get(0, 1)), fp("22"), tight(), "matmul[0,1]");
    assert_fp(sv_to_fp(c.get(1, 0)), fp("43"), tight(), "matmul[1,0]");
    assert_fp(sv_to_fp(c.get(1, 1)), fp("50"), tight(), "matmul[1,1]");
}

// ============================================================================
// Decimal domain — financial-grade
// ============================================================================

#[test]
fn test_decimal_domain_matrix_creation() {
    let m = DomainMatrix::from_strings(2, 2, &["0.10", "0.20", "0.30", "0.40"]).unwrap();
    assert_eq!(m.rows(), 2);
    assert!(m.is_uniform_domain());
    assert_eq!(m.dominant_domain(), Some(DomainType::Decimal));
}

#[test]
fn test_decimal_domain_matrix_add() {
    // Decimal addition should be exact (0 ULP)
    let a = DomainMatrix::from_strings(2, 2, &["0.10", "0.20", "0.30", "0.40"]).unwrap();
    let b = DomainMatrix::from_strings(2, 2, &["0.01", "0.02", "0.03", "0.04"]).unwrap();
    let c = a.add(&b).unwrap();
    // 0.10 + 0.01 = 0.11 (exact in decimal domain)
    let r = c.to_fixed_matrix().unwrap();
    assert_fp(r.get(0, 0), fp("0.11"), tight(), "decimal_add[0,0]");
    assert_fp(r.get(1, 1), fp("0.44"), tight(), "decimal_add[1,1]");
}

#[test]
fn test_decimal_domain_matrix_sub() {
    let a = DomainMatrix::from_strings(2, 2, &["0.50", "0.60", "0.70", "0.80"]).unwrap();
    let b = DomainMatrix::from_strings(2, 2, &["0.10", "0.20", "0.30", "0.40"]).unwrap();
    let c = a.sub(&b).unwrap();
    let r = c.to_fixed_matrix().unwrap();
    assert_fp(r.get(0, 0), fp("0.40"), tight(), "decimal_sub[0,0]");
    assert_fp(r.get(1, 1), fp("0.40"), tight(), "decimal_sub[1,1]");
}

#[test]
fn test_decimal_identity_matmul() {
    // Decimal identity * decimal matrix = same matrix
    let id = DomainMatrix::from_strings(2, 2, &["1.00", "0.00", "0.00", "1.00"]).unwrap();
    let v = DomainMatrix::from_strings(2, 1, &["0.10", "0.20"]).unwrap();
    let result = id.mat_mul(&v).unwrap();
    let r = result.to_fixed_matrix().unwrap();
    assert_fp(r.get(0, 0), fp("0.10"), tight(), "decimal_id_mul[0]");
    assert_fp(r.get(1, 0), fp("0.20"), tight(), "decimal_id_mul[1]");
}

// ============================================================================
// Cross-domain operations (mixed decimal + binary)
// ============================================================================

#[test]
fn test_cross_domain_add() {
    // Binary + Decimal — should route through rational and produce correct result
    let a = DomainMatrix::from_strings(2, 2, &["1", "2", "3", "4"]).unwrap();
    let b = DomainMatrix::from_strings(2, 2, &["0.50", "0.50", "0.50", "0.50"]).unwrap();
    let c = a.add(&b).unwrap();
    let r = c.to_fixed_matrix().unwrap();
    assert_fp(r.get(0, 0), fp("1.5"), tight(), "cross_add[0,0]");
    assert_fp(r.get(1, 1), fp("4.5"), tight(), "cross_add[1,1]");
}

#[test]
fn test_cross_domain_matmul() {
    // Binary matrix * Decimal matrix
    let a = DomainMatrix::from_strings(2, 2, &["2", "0", "0", "3"]).unwrap();
    let b = DomainMatrix::from_strings(2, 2, &["0.10", "0.20", "0.30", "0.40"]).unwrap();
    let c = a.mat_mul(&b).unwrap();
    let r = c.to_fixed_matrix().unwrap();
    // [[2,0],[0,3]] * [[0.1,0.2],[0.3,0.4]] = [[0.2,0.4],[0.9,1.2]]
    assert_fp(r.get(0, 0), fp("0.2"), tight(), "cross_mul[0,0]");
    assert_fp(r.get(0, 1), fp("0.4"), tight(), "cross_mul[0,1]");
    assert_fp(r.get(1, 0), fp("0.9"), tight(), "cross_mul[1,0]");
    assert_fp(r.get(1, 1), fp("1.2"), tight(), "cross_mul[1,1]");
}

// ============================================================================
// Symbolic domain (exact rational)
// ============================================================================

#[test]
fn test_symbolic_domain_matrix() {
    // Rational entries: 1/3, 2/3 — exact symbolic representation
    let m = DomainMatrix::from_strings(2, 2, &["1/3", "2/3", "1/6", "5/6"]).unwrap();
    assert_eq!(m.rows(), 2);
    assert!(m.is_uniform_domain());
    assert_eq!(m.dominant_domain(), Some(DomainType::Symbolic));
}

#[test]
fn test_symbolic_domain_add() {
    // 1/3 + 1/6 = 1/2 (exact)
    let a = DomainMatrix::from_strings(1, 1, &["1/3"]).unwrap();
    let b = DomainMatrix::from_strings(1, 1, &["1/6"]).unwrap();
    let c = a.add(&b).unwrap();
    let r = c.to_fixed_matrix().unwrap();
    assert_fp(r.get(0, 0), fp("0.5"), tight(), "symbolic_1/3+1/6=1/2");
}

// ============================================================================
// Utility operations
// ============================================================================

#[test]
fn test_domain_matrix_transpose() {
    let m = DomainMatrix::from_strings(2, 3, &["1", "2", "3", "4", "5", "6"]).unwrap();
    let mt = m.transpose();
    assert_eq!(mt.rows(), 3);
    assert_eq!(mt.cols(), 2);
}

#[test]
fn test_domain_matrix_neg() {
    let m = DomainMatrix::from_strings(2, 2, &["1", "2", "3", "4"]).unwrap();
    let neg = m.neg().unwrap();
    let r = neg.to_fixed_matrix().unwrap();
    assert_fp(r.get(0, 0), fp("-1"), tight(), "neg[0,0]");
    assert_fp(r.get(1, 1), fp("-4"), tight(), "neg[1,1]");
}

#[test]
fn test_domain_matrix_scalar_mul() {
    let m = DomainMatrix::from_strings(2, 2, &["1", "2", "3", "4"]).unwrap();
    let s = evaluate(&gmath("3")).unwrap();
    let scaled = m.scalar_mul(&s).unwrap();
    let r = scaled.to_fixed_matrix().unwrap();
    assert_fp(r.get(0, 0), fp("3"), tight(), "scalar_mul[0,0]");
    assert_fp(r.get(1, 1), fp("12"), tight(), "scalar_mul[1,1]");
}

#[test]
fn test_domain_matrix_trace() {
    let m = DomainMatrix::from_strings(3, 3, &[
        "1", "0", "0",
        "0", "2", "0",
        "0", "0", "3",
    ]).unwrap();
    let tr = m.trace().unwrap();
    assert_fp(sv_to_fp(&tr), fp("6"), tight(), "trace");
}

#[test]
fn test_domain_matrix_to_from_fixed() {
    let fm = FixedMatrix::from_fn(2, 2, |r, c| fp(&format!("{}", (r * 2 + c + 1))));
    let dm = DomainMatrix::from_fixed_matrix(&fm);
    let roundtrip = dm.to_fixed_matrix().unwrap();
    for r in 0..2 {
        for c in 0..2 {
            assert_fp(roundtrip.get(r, c), fm.get(r, c), tight(), &format!("roundtrip[{r},{c}]"));
        }
    }
}
