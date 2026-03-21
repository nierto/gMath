//! ULP measurement for L1D matrix functions and L3A manifold operations.

use g_math::fixed_point::{FixedPoint, FixedVector, FixedMatrix};
use g_math::fixed_point::imperative::matrix_functions::*;
use g_math::fixed_point::imperative::manifold::*;

fn fp(s: &str) -> FixedPoint {
    if s.starts_with('-') { -FixedPoint::from_str(&s[1..]) }
    else { FixedPoint::from_str(s) }
}

fn ulp_diff(a: FixedPoint, b: FixedPoint) -> u64 {
    let diff = a - b;
    let raw = diff.raw();
    #[cfg(table_format = "q64_64")]
    { raw.unsigned_abs() as u64 }
    #[cfg(any(table_format = "q128_128", table_format = "q256_256"))]
    { let abs_raw = if raw.is_negative() { -raw } else { raw }; abs_raw.as_i128() as u64 }
}

fn report(name: &str, got: FixedPoint, reference: FixedPoint) -> u64 {
    let ulp = ulp_diff(got, reference);
    println!("  {:<45} {:>8} ULP", name, ulp);
    ulp
}

#[test]
fn test_l1d_l3a_ulp_report() {
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║    L1D/L3A ULP MEASUREMENT (vs mpmath 50-digit)            ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let mut max_ulp = 0u64;

    // ── Matrix Exp: rotation matrix ──
    println!("── matrix_exp: [[0,1],[-1,0]] → rotation ──");
    {
        let a = FixedMatrix::from_slice(2, 2, &[fp("0"), fp("1"), fp("-1"), fp("0")]);
        let r = matrix_exp(&a).unwrap();
        max_ulp = max_ulp.max(report("exp_rot[0][0] = cos(1)", r.get(0, 0), fp("0.5403023058681397174")));
        max_ulp = max_ulp.max(report("exp_rot[0][1] = sin(1)", r.get(0, 1), fp("0.8414709848078965066")));
        max_ulp = max_ulp.max(report("exp_rot[1][0] = -sin(1)", r.get(1, 0), fp("-0.8414709848078965066")));
        max_ulp = max_ulp.max(report("exp_rot[1][1] = cos(1)", r.get(1, 1), fp("0.5403023058681397174")));
    }
    println!();

    // ── Matrix Exp: diagonal ──
    println!("── matrix_exp: [[1,0],[0,2]] → diag(e, e²) ──");
    {
        let a = FixedMatrix::from_slice(2, 2, &[fp("1"), fp("0"), fp("0"), fp("2")]);
        let r = matrix_exp(&a).unwrap();
        max_ulp = max_ulp.max(report("exp_diag[0][0] = e", r.get(0, 0), fp("2.718281828459045235")));
        max_ulp = max_ulp.max(report("exp_diag[1][1] = e²", r.get(1, 1), fp("7.389056098930650227")));
        max_ulp = max_ulp.max(report("exp_diag[0][1] = 0", r.get(0, 1), fp("0")));
    }
    println!();

    // ── Matrix Sqrt ──
    println!("── matrix_sqrt: [[2,1],[1,2]] ──");
    {
        let a = FixedMatrix::from_slice(2, 2, &[fp("2"), fp("1"), fp("1"), fp("2")]);
        let s = matrix_sqrt(&a).unwrap();
        max_ulp = max_ulp.max(report("sqrt[0][0]", s.get(0, 0), fp("1.366025403784438646")));
        max_ulp = max_ulp.max(report("sqrt[0][1]", s.get(0, 1), fp("0.366025403784438646")));
        max_ulp = max_ulp.max(report("sqrt[1][0]", s.get(1, 0), fp("0.366025403784438646")));
        // Reconstruction: S² should equal A
        let s_sq = &s * &s;
        for r in 0..2 {
            for c in 0..2 {
                max_ulp = max_ulp.max(report(
                    &format!("sqrt²[{}][{}] vs A", r, c),
                    s_sq.get(r, c), a.get(r, c)));
            }
        }
    }
    println!();

    // ── Sphere distance ──
    println!("── Sphere S² distance ──");
    {
        let s = Sphere { dim: 2 };
        let p = FixedVector::from_slice(&[fp("1"), fp("0"), fp("0")]);
        let q = FixedVector::from_slice(&[fp("0"), fp("1"), fp("0")]);
        let d = s.distance(&p, &q).unwrap();
        max_ulp = max_ulp.max(report("d([1,0,0],[0,1,0]) = π/2", d, fp("1.5707963267948966192")));

        let d_same = s.distance(&p, &p).unwrap();
        max_ulp = max_ulp.max(report("d(p, p) = 0", d_same, fp("0")));
    }
    println!();

    // ── Hyperbolic distance ──
    println!("── Hyperbolic H¹ distance ──");
    {
        let h = HyperbolicSpace { dim: 1 };
        let p = FixedVector::from_slice(&[fp("1"), fp("0")]);
        let q = FixedVector::from_slice(&[fp("1.5430806348152437784"), fp("1.1752011936438014568")]);
        let d = h.distance(&p, &q).unwrap();
        max_ulp = max_ulp.max(report("d(origin, cosh1/sinh1) = 1", d, fp("1")));
    }
    println!();

    // ── Exp/Log roundtrip ──
    println!("── exp(log(A)) = A roundtrip ──");
    {
        let a = FixedMatrix::from_slice(2, 2, &[fp("2"), fp("1"), fp("1"), fp("2")]);
        let log_a = matrix_log(&a).unwrap();
        let exp_log_a = matrix_exp(&log_a).unwrap();
        for r in 0..2 {
            for c in 0..2 {
                max_ulp = max_ulp.max(report(
                    &format!("exp(log(A))[{}][{}]", r, c),
                    exp_log_a.get(r, c), a.get(r, c)));
            }
        }
    }
    println!();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  MAX ULP ACROSS ALL L1D/L3A: {:>8}                      ║", max_ulp);
    println!("╚══════════════════════════════════════════════════════════════╝");
}
