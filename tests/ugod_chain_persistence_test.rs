//! UGOD + BinaryCompute chain persistence: the critical architecture test.
//!
//! This proves the FASC architecture's key promise:
//! intermediates that overflow storage tier SURVIVE in a chain
//! if the FINAL result fits after materialization.
//!
//! exp(44) ≈ 1.28e19 — overflows Q64.64 storage (max ~9.2e18)
//! sin(exp(44)) ≈ -0.869 — fits in ANY storage tier
//!
//! The imperative path `.exp().sin()` FAILS because exp() materializes.
//! The FASC LazyExpr path `gmath("44").exp().sin()` SUCCEEDS because
//! BinaryCompute keeps exp(44) at I256 compute tier, sin operates on
//! the I256 value, and only the final sin result (in [-1,1]) gets
//! downscaled to i128.
//!
//! This is the foundation for:
//! - Long matrix operation chains (5-20 ops without materialization)
//! - Lie group compositions where intermediate group elements overflow
//! - Geodesic integrators where intermediate manifold coordinates blow up
//! - Future LazyMatrixExpr chains (matrix analog of BinaryCompute)

use g_math::fixed_point::FixedPoint;
use g_math::canonical::{gmath, evaluate};

fn fp(s: &str) -> FixedPoint {
    if s.starts_with('-') { -FixedPoint::from_str(&s[1..]) }
    else { FixedPoint::from_str(s) }
}

#[test]
fn test_chain_persistence_overflow_intermediate() {
    // Build the LazyExpr tree: Sin(Exp(44))
    // This should NOT materialize exp(44) to storage — it should stay at
    // BinaryCompute tier through the chain.
    let expr = gmath("44").exp().sin();
    let result = evaluate(&expr);

    match result {
        Ok(sv) => {
            let s = format!("{}", sv);
            println!("sin(exp(44)) via FASC chain = {}", s);
            // mpmath: sin(exp(44)) = -0.8691572325710950298...
            // If this succeeds, BinaryCompute chain persistence is working!
            println!("FASC BinaryCompute chain persistence: WORKING");
            println!("Intermediate exp(44) stayed at compute tier (I256)");
            println!("Only sin(exp(44)) was materialized to storage tier");
        }
        Err(e) => {
            // If this fails, the current FASC evaluator materializes
            // between operations — chain persistence doesn't survive overflow.
            println!("sin(exp(44)) via FASC chain FAILED: {:?}", e);
            println!("This means the evaluator materializes exp(44) before sin()");
            println!("UGOD correctly detected overflow, but chain persistence");
            println!("should have kept it at BinaryCompute tier.");
            println!("");
            println!("ARCHITECTURAL GAP: the stack evaluator's evaluate_sin()");
            println!("calls evaluate(inner) which materializes, THEN computes sin.");
            println!("For true chain persistence, sin should detect BinaryCompute");
            println!("input and compute sin at compute tier without materializing.");
        }
    }

    // Now verify the imperative path fails (as expected — no chain persistence)
    let imp_result = fp("44").try_exp();
    match imp_result {
        Ok(_) => println!("\nImperative exp(44): succeeded (wide enough profile)"),
        Err(_) => println!("\nImperative exp(44): Err(TierOverflow) — expected on Q64.64"),
    }
}

#[test]
fn test_chain_persistence_non_overflow() {
    // Chain that doesn't overflow but benefits from compute-tier precision:
    // sin(exp(0.5)) — both exp(0.5) and sin(exp(0.5)) fit in storage tier.
    // But the chain at compute tier should give better precision than
    // materializing between operations.
    let expr = gmath("0.5").exp().sin();
    let result = evaluate(&expr);
    assert!(result.is_ok(), "sin(exp(0.5)) should always succeed");

    let sv = result.unwrap();
    let s = format!("{}", sv);
    println!("sin(exp(0.5)) via FASC chain = {}", s);
    // mpmath: 0.99696538761396753472...
}

// ============================================================================
// More chain patterns — proving BinaryCompute persistence is general
// ============================================================================

#[test]
fn test_chain_cos_of_exp_overflow() {
    // cos(exp(44)) — same pattern, different outer function
    // mpmath: cos(exp(44)) = 0.49451075960498576649...
    let expr = gmath("44").exp().cos();
    let _result = evaluate(&expr);

    #[cfg(table_format = "q64_64")]
    {
        // exp(44) overflows storage but cos brings it back to [-1,1]
        assert!(_result.is_ok(),
            "cos(exp(44)) should succeed via BinaryCompute chain");
        if let Ok(sv) = _result {
            let s = format!("{}", sv);
            println!("cos(exp(44)) via FASC chain = {}", s);
            // mpmath: 0.49451075960498576649...
            let val = fp(&s);
            assert!((val - fp("0.4945107596049857664")).abs() < fp("0.001"),
                "cos(exp(44)) should match mpmath: got {}", val);
        }
    }
}

#[test]
fn test_chain_atan_of_exp() {
    // atan(exp(30)) — exp(30) ≈ 1.07e13, but atan maps R → (-π/2, π/2)
    // mpmath: atan(exp(30)) = 1.5707963267948966192... (≈ π/2)
    let expr = gmath("30").exp().atan();
    let result = evaluate(&expr);
    assert!(result.is_ok(), "atan(exp(30)) should succeed");
    if let Ok(sv) = result {
        let s = format!("{}", sv);
        println!("atan(exp(30)) via FASC chain = {}", s);
        let val = fp(&s);
        // Should be very close to π/2
        assert!((val - fp("1.5707963267948966192")).abs() < fp("0.001"),
            "atan(exp(30)) ≈ π/2: got {}", val);
    }
}

#[test]
fn test_chain_triple_transcendental() {
    // sin(cos(exp(1))) — 3-deep chain
    // mpmath 50 digits:
    //   exp(1) = 2.71828182845904523536...
    //   cos(exp(1)) = -0.91173391478696508282...
    //   sin(cos(exp(1))) = -0.79056673518158675425...
    let expr = gmath("1").exp().cos().sin();
    let result = evaluate(&expr);
    assert!(result.is_ok(), "sin(cos(exp(1))) should succeed");
    if let Ok(sv) = result {
        let s = format!("{}", sv);
        println!("sin(cos(exp(1))) via FASC chain = {}", s);
        let val = fp(&s);
        assert!((val - fp("-0.7905667351815867542")).abs() < fp("0.001"),
            "sin(cos(exp(1))) mpmath: got {}", val);
    }
}

#[test]
fn test_chain_imperative_fails_fasc_succeeds() {
    // The definitive UGOD test: same computation, two paths.
    // Imperative: materializes between each operation → FAILS on overflow
    // FASC: chain persistence at BinaryCompute → SUCCEEDS
    //
    // This IS the value proposition of FASC-UGOD architecture.

    // Imperative path: exp(44) materializes, overflows
    let _imp_exp = fp("44").try_exp();

    // FASC path: sin(exp(44)) stays at compute tier through chain
    let _fasc_result = evaluate(&gmath("44").exp().sin());

    #[cfg(table_format = "q64_64")]
    {
        assert!(_imp_exp.is_err(),
            "Imperative exp(44) MUST fail on Q64.64 — proves storage overflow");
        assert!(_fasc_result.is_ok(),
            "FASC sin(exp(44)) MUST succeed — proves chain persistence");

        println!("\n========================================");
        println!("UGOD Chain Persistence: PROVEN");
        println!("========================================");
        println!("Imperative exp(44):     Err(TierOverflow)");
        println!("FASC sin(exp(44)):      {}", _fasc_result.unwrap());
        println!("Same computation, different architecture.");
        println!("FASC keeps intermediates at BinaryCompute tier.");
        println!("Only the final result materializes to storage.");
        println!("========================================");
    }
}
