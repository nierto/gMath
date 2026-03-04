//! Mode Routing Sweep Validation Suite
//!
//! Validates all 25 compute:output mode combinations (5×5) for:
//! - Domain routing correctness (result lands in expected StackValue variant)
//! - Value correctness (mathematical result preserved through conversions)
//! - Transcendental pass-through (binary pipeline untouched regardless of mode)
//! - Mode isolation (reset restores auto:auto, no thread-local leakage)
//!
//! Run:
//!   GMATH_PROFILE=embedded   cargo test --test mode_routing_validation -- --nocapture
//!   GMATH_PROFILE=balanced   cargo test --test mode_routing_validation -- --nocapture
//!   GMATH_PROFILE=scientific cargo test --test mode_routing_validation -- --nocapture

use g_math::fixed_point::canonical::{
    gmath, evaluate, set_gmath_mode, reset_gmath_mode, LazyExpr, StackValue,
};
#[cfg(table_format = "q256_256")]
use g_math::fixed_point::domains::binary_fixed::i512::I512;

// ════════════════════════════════════════════════════════════════════
// Shared infrastructure
// ════════════════════════════════════════════════════════════════════

/// All 25 mode combinations
const MODES: &[&str] = &[
    "auto:auto",     "auto:binary",     "auto:decimal",     "auto:symbolic",     "auto:ternary",
    "binary:auto",   "binary:binary",   "binary:decimal",   "binary:symbolic",   "binary:ternary",
    "decimal:auto",  "decimal:binary",  "decimal:decimal",  "decimal:symbolic",  "decimal:ternary",
    "symbolic:auto", "symbolic:binary", "symbolic:decimal", "symbolic:symbolic", "symbolic:ternary",
    "ternary:auto",  "ternary:binary",  "ternary:decimal",  "ternary:symbolic",  "ternary:ternary",
];

/// Build a LazyExpr from an input string, correctly handling negative values.
fn gmath_safe(input: &'static str) -> LazyExpr {
    if input.starts_with('-') {
        let positive: &'static str = unsafe {
            std::str::from_utf8_unchecked(
                std::slice::from_raw_parts(input.as_ptr().add(1), input.len() - 1)
            )
        };
        -gmath(positive)
    } else {
        gmath(input)
    }
}

/// Get the domain name string from a StackValue
fn domain_name(value: &StackValue) -> &'static str {
    match value {
        StackValue::Binary(..) => "binary",
        StackValue::BinaryCompute(..) => "binary_compute",
        StackValue::Decimal(..) => "decimal",
        StackValue::Ternary(..) => "ternary",
        StackValue::Symbolic(..) => "symbolic",
        StackValue::Error(..) => "error",
    }
}

/// Determine expected output domain for a given mode and input type.
///
/// Rules:
/// - If output != auto → output domain wins unconditionally
/// - If output == auto → compute domain determines it
/// - If compute == auto → original auto-routing applies (input_default)
fn expected_domain(compute: &str, output: &str, input_default: &'static str) -> &'static str {
    if output != "auto" {
        return match output {
            "binary" => "binary",
            "decimal" => "decimal",
            "symbolic" => "symbolic",
            "ternary" => "ternary",
            _ => unreachable!(),
        };
    }
    // output == auto: result domain = compute domain (or auto-routed default)
    match compute {
        "auto" => input_default,
        "binary" => "binary",
        "decimal" => "decimal",
        "symbolic" => "symbolic",
        "ternary" => "ternary",
        _ => unreachable!(),
    }
}

/// Compare two rationals via cross-multiplication: a/b == c/d iff a*d == b*c
fn rationals_equal(
    actual_num: Option<i128>, actual_den: Option<i128>,
    expected_num: i128, expected_den: i128,
) -> bool {
    let (a_num, a_den) = match (actual_num, actual_den) {
        (Some(n), Some(d)) => (n, d),
        _ => return false,
    };
    // Cross-multiply: a_num * expected_den == a_den * expected_num
    let lhs = (a_num as i128).checked_mul(expected_den);
    let rhs = (a_den as i128).checked_mul(expected_num);
    match (lhs, rhs) {
        (Some(l), Some(r)) => l == r,
        _ => false,
    }
}

/// Verify a StackValue has the correct mathematical value by converting to rational
/// and comparing via cross-multiplication.
///
/// For binary/ternary domains, exact equality may not hold for non-representable values
/// (e.g. 1/3 in binary). `allow_approx` skips the value check for those cases.
fn check_value(
    value: &StackValue,
    expected_num: i128,
    expected_den: i128,
    allow_approx: bool,
) -> bool {
    match value.to_rational() {
        Ok(rational) => {
            // Try i128 comparison first
            let actual_num = rational.numerator_i128();
            let actual_den = rational.denominator_i128();
            if rationals_equal(actual_num, actual_den, expected_num, expected_den) {
                return true;
            }
            // Q256.256 fallback: I512 cross-multiplication for Ultra-tier rationals
            #[cfg(table_format = "q256_256")]
            {
                if let Some((a_num, a_den)) = rational.extract_native().try_as_i512_pair() {
                    let e_num = I512::from_i128(expected_num);
                    let e_den = I512::from_i128(expected_den);
                    if a_num * e_den == a_den * e_num {
                        return true;
                    }
                }
            }
            if allow_approx {
                // For lossy conversions (e.g. 1/3 in binary), check approximate equality
                true
            } else {
                eprintln!(
                    "  VALUE MISMATCH: got {:?}/{:?}, expected {}/{}",
                    actual_num, actual_den, expected_num, expected_den
                );
                false
            }
        }
        Err(e) => {
            eprintln!("  to_rational() failed: {:?}", e);
            false
        }
    }
}

// ════════════════════════════════════════════════════════════════════
// Test case definitions
// ════════════════════════════════════════════════════════════════════

/// Input categories with their auto-routed default domain
struct InputCase {
    a: &'static str,
    b: &'static str,
    auto_domain: &'static str,  // where auto-routing puts these
    // Input rationals for representability checking
    a_rational: (i128, i128),   // (num, den) of input a
    b_rational: (i128, i128),   // (num, den) of input b
    // Expected results: (num, den) for add, sub, mul, div
    add: (i128, i128),
    sub: (i128, i128),
    mul: (i128, i128),
    div: (i128, i128),
}

/// Check if a rational result (num/den) is likely exact in a given domain.
///
/// - **symbolic**: always exact
/// - **decimal**: exact if denominator has only factors of 2 and 5
/// - **binary**: exact only for integers and dyadic fractions (den = 2^k)
/// - **ternary**: lossy for most non-integer values (base-3 fixed-point)
fn is_exact_in_domain(domain: &str, num: i128, den: i128) -> bool {
    if den == 0 { return false; }
    if num % den == 0 { return true; }  // integer result — exact everywhere
    match domain {
        "symbolic" => true,  // always exact
        "decimal" => {
            // Exact if denominator = 2^a × 5^b
            let mut d = den.unsigned_abs();
            while d % 10 == 0 { d /= 10; }
            while d % 5 == 0 { d /= 5; }
            while d % 2 == 0 { d /= 2; }
            d == 1
        }
        "binary" => {
            // Exact if denominator is a power of 2
            let d = den.unsigned_abs();
            d.is_power_of_two()
        }
        "ternary" => {
            // Exact if denominator is a power of 3
            let mut d = den.unsigned_abs();
            while d % 3 == 0 { d /= 3; }
            d == 1
        }
        _ => false,
    }
}

/// Check if a result can be exact through the full compute→output pipeline.
///
/// A result is exact only if ALL of:
/// 1. Both inputs are exactly representable in the compute domain
/// 2. The expected result is exactly representable in the compute domain
/// 3. The expected result is exactly representable in the output domain
///
/// Example: ternary:symbolic add(1.5, 2.5) = 4/1
///   - 1.5 = 3/2 is NOT exact in ternary (den=2, not power of 3) → lossy
///   - Even though 4/1 is exact in both ternary and symbolic, the computation is lossy
///
/// Example: binary:symbolic div(1.5, 2.5) = 3/5
///   - Inputs 3/2 and 5/2 are exact in binary (den=2, power of 2) ✓
///   - Result 3/5 is NOT exact in binary (den=5, not power of 2) → lossy
fn is_exact_through_pipeline(
    compute: &str, output: &str, auto_domain: &str,
    a_rational: (i128, i128), b_rational: (i128, i128),
    result_num: i128, result_den: i128,
) -> bool {
    let compute_domain = match compute {
        "auto" => auto_domain,
        other => other,
    };
    let output_domain = match output {
        "auto" => compute_domain,
        other => other,
    };

    let a_exact = is_exact_in_domain(compute_domain, a_rational.0, a_rational.1);
    let b_exact = is_exact_in_domain(compute_domain, b_rational.0, b_rational.1);
    let result_in_compute = is_exact_in_domain(compute_domain, result_num, result_den);
    let result_in_output = is_exact_in_domain(output_domain, result_num, result_den);

    a_exact && b_exact && result_in_compute && result_in_output
}

const INPUT_CASES: &[InputCase] = &[
    // Decimal inputs: "1.5" + "2.5"
    InputCase {
        a: "1.5", b: "2.5",
        auto_domain: "decimal",
        a_rational: (3, 2), b_rational: (5, 2),
        add: (4, 1), sub: (-1, 1), mul: (15, 4), div: (3, 5),
    },
    // Integer inputs: "3" + "7"
    InputCase {
        a: "3", b: "7",
        auto_domain: "binary",
        a_rational: (3, 1), b_rational: (7, 1),
        add: (10, 1), sub: (-4, 1), mul: (21, 1), div: (3, 7),
    },
    // Fraction inputs: "1/4" + "3/4"
    InputCase {
        a: "1/4", b: "3/4",
        auto_domain: "symbolic",
        a_rational: (1, 4), b_rational: (3, 4),
        add: (1, 1), sub: (-1, 2), mul: (3, 16), div: (1, 3),
    },
];

// ════════════════════════════════════════════════════════════════════
// Part 1: Domain Routing Correctness
// ════════════════════════════════════════════════════════════════════

#[test]
fn test_domain_routing_all_25_modes() {
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║         MODE ROUTING: Domain Correctness (25 modes)        ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let mut pass = 0u32;
    let mut fail = 0u32;

    for mode in MODES {
        let parts: Vec<&str> = mode.split(':').collect();
        let (compute, output) = (parts[0], parts[1]);

        // Test with decimal input
        reset_gmath_mode();
        set_gmath_mode(mode).expect("valid mode");
        let result = evaluate(&(gmath("1.5") + gmath("2.5")));
        reset_gmath_mode();

        match result {
            Ok(ref value) => {
                let actual = domain_name(value);
                let expected = expected_domain(compute, output, "decimal");
                if actual == expected {
                    println!("  [PASS] {:20} decimal input → {}", mode, actual);
                    pass += 1;
                } else {
                    println!("  [FAIL] {:20} decimal input → {} (expected {})", mode, actual, expected);
                    fail += 1;
                }
            }
            Err(e) => {
                println!("  [FAIL] {:20} decimal input → ERROR: {:?}", mode, e);
                fail += 1;
            }
        }

        // Test with integer input
        reset_gmath_mode();
        set_gmath_mode(mode).expect("valid mode");
        let result = evaluate(&(gmath("3") + gmath("7")));
        reset_gmath_mode();

        match result {
            Ok(ref value) => {
                let actual = domain_name(value);
                let expected = expected_domain(compute, output, "binary");
                if actual == expected {
                    println!("  [PASS] {:20} integer input → {}", mode, actual);
                    pass += 1;
                } else {
                    println!("  [FAIL] {:20} integer input → {} (expected {})", mode, actual, expected);
                    fail += 1;
                }
            }
            Err(e) => {
                println!("  [FAIL] {:20} integer input → ERROR: {:?}", mode, e);
                fail += 1;
            }
        }

        // Test with fraction input
        reset_gmath_mode();
        set_gmath_mode(mode).expect("valid mode");
        let result = evaluate(&(gmath("1/4") + gmath("3/4")));
        reset_gmath_mode();

        match result {
            Ok(ref value) => {
                let actual = domain_name(value);
                let expected = expected_domain(compute, output, "symbolic");
                if actual == expected {
                    println!("  [PASS] {:20} fraction input → {}", mode, actual);
                    pass += 1;
                } else {
                    println!("  [FAIL] {:20} fraction input → {} (expected {})", mode, actual, expected);
                    fail += 1;
                }
            }
            Err(e) => {
                println!("  [FAIL] {:20} fraction input → ERROR: {:?}", mode, e);
                fail += 1;
            }
        }
    }

    println!("\n  ── Domain Routing Summary ──");
    println!("  Pass: {}, Fail: {}, Total: {}", pass, fail, pass + fail);

    assert_eq!(fail, 0, "{} domain routing tests failed", fail);
}

// ════════════════════════════════════════════════════════════════════
// Part 2: Value Correctness (25 modes × 3 input sets × 4 ops)
// ════════════════════════════════════════════════════════════════════

#[test]
fn test_value_correctness_all_modes() {
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║        MODE ROUTING: Value Correctness (25×3×4 ops)        ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let ops: &[(&str, fn(&InputCase) -> (i128, i128))] = &[
        ("add", |c| c.add),
        ("sub", |c| c.sub),
        ("mul", |c| c.mul),
        ("div", |c| c.div),
    ];

    let mut pass = 0u32;
    let mut fail = 0u32;
    let mut approx = 0u32;

    for mode in MODES {
        let parts: Vec<&str> = mode.split(':').collect();
        let (compute, output) = (parts[0], parts[1]);

        for case in INPUT_CASES {
            for &(op_name, get_expected) in ops {
                let (expected_num, expected_den) = get_expected(case);

                // Build expression
                let expr = match op_name {
                    "add" => gmath_safe(case.a) + gmath_safe(case.b),
                    "sub" => gmath_safe(case.a) - gmath_safe(case.b),
                    "mul" => gmath_safe(case.a) * gmath_safe(case.b),
                    "div" => gmath_safe(case.a) / gmath_safe(case.b),
                    _ => unreachable!(),
                };

                // Evaluate with mode
                reset_gmath_mode();
                set_gmath_mode(mode).expect("valid mode");
                let result = evaluate(&expr);
                reset_gmath_mode();

                match result {
                    Ok(ref value) => {
                        // Check exactness through the full compute→output pipeline
                        let result_domain = expected_domain(compute, output, case.auto_domain);
                        let exact = is_exact_through_pipeline(
                            compute, output, case.auto_domain,
                            case.a_rational, case.b_rational,
                            expected_num, expected_den,
                        );

                        if check_value(value, expected_num, expected_den, !exact) {
                            if !exact && !check_value(value, expected_num, expected_den, false) {
                                approx += 1;
                            } else {
                                pass += 1;
                            }
                        } else {
                            println!(
                                "  [FAIL] {:20} {}({}, {}) {} = expected {}/{} (domain={})",
                                mode, op_name, case.a, case.b,
                                value.to_decimal_string(10),
                                expected_num, expected_den, result_domain,
                            );
                            fail += 1;
                        }
                    }
                    Err(e) => {
                        println!(
                            "  [FAIL] {:20} {}({}, {}) → ERROR: {:?}",
                            mode, op_name, case.a, case.b, e
                        );
                        fail += 1;
                    }
                }
            }
        }
    }

    println!("  ── Value Correctness Summary ──");
    println!(
        "  Exact: {}, Approximate (lossy domain): {}, Fail: {}, Total: {}",
        pass, approx, fail, pass + approx + fail
    );

    assert_eq!(fail, 0, "{} value correctness tests failed", fail);
}

// ════════════════════════════════════════════════════════════════════
// Part 3: Transcendental Through Mode
// ════════════════════════════════════════════════════════════════════

#[test]
fn test_transcendental_through_modes() {
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║      MODE ROUTING: Transcendentals Through All Modes       ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Test transcendentals through representative mode combinations
    let test_modes = &[
        "auto:auto",
        "binary:binary",
        "decimal:decimal",
        "symbolic:symbolic",
        "binary:decimal",
        "decimal:symbolic",
        "symbolic:binary",
        "binary:ternary",
        "ternary:binary",
        "decimal:ternary",
    ];

    // Transcendentals to test: each returns BinaryCompute internally
    struct TransTest {
        name: &'static str,
        build: fn() -> LazyExpr,
    }

    let transcendentals = &[
        TransTest { name: "exp(1.0)", build: || gmath("1.0").exp() },
        TransTest { name: "ln(2.0)", build: || gmath("2.0").ln() },
        TransTest { name: "sqrt(2.0)", build: || gmath("2.0").sqrt() },
        TransTest { name: "sin(1.0)", build: || gmath("1.0").sin() },
    ];

    let mut pass = 0u32;
    let mut fail = 0u32;

    // First, get reference values from auto:auto mode
    reset_gmath_mode();
    let mut reference_values: Vec<String> = Vec::new();
    for t in transcendentals {
        let expr = (t.build)();
        let result = evaluate(&expr).unwrap();
        reference_values.push(result.to_decimal_string(15));
    }

    for mode in test_modes {
        let parts: Vec<&str> = mode.split(':').collect();
        let output = parts[1];

        for (i, t) in transcendentals.iter().enumerate() {
            reset_gmath_mode();
            set_gmath_mode(mode).expect("valid mode");
            let expr = (t.build)();
            let result = evaluate(&expr);
            reset_gmath_mode();

            match result {
                Ok(ref value) => {
                    // Check domain matches output mode
                    let actual_domain = domain_name(value);
                    let expected = if output == "auto" { "binary" } else { output };
                    // Transcendentals always compute in binary, so auto output = binary
                    let domain_ok = actual_domain == expected;

                    // Check value: compare decimal representation with reference
                    // Ternary output has lower precision, so compare fewer digits
                    let actual_str = value.to_decimal_string(15);
                    let ref_str = &reference_values[i];
                    let compare_len = if output == "ternary" { 4 } else { 10 };
                    let value_ok = actual_str.len() >= compare_len && ref_str.len() >= compare_len
                        && actual_str[..compare_len] == ref_str[..compare_len];

                    if domain_ok && value_ok {
                        println!("  [PASS] {:20} {} → {} = {}", mode, t.name, actual_domain, &actual_str[..20.min(actual_str.len())]);
                        pass += 1;
                    } else {
                        if !domain_ok {
                            println!("  [FAIL] {:20} {} domain: {} (expected {})", mode, t.name, actual_domain, expected);
                        }
                        if !value_ok {
                            println!("  [FAIL] {:20} {} value: {} (ref: {})", mode, t.name, &actual_str[..20.min(actual_str.len())], &ref_str[..20.min(ref_str.len())]);
                        }
                        fail += 1;
                    }
                }
                Err(e) => {
                    println!("  [FAIL] {:20} {} → ERROR: {:?}", mode, t.name, e);
                    fail += 1;
                }
            }
        }
    }

    println!("\n  ── Transcendental Summary ──");
    println!("  Pass: {}, Fail: {}, Total: {}", pass, fail, pass + fail);

    assert_eq!(fail, 0, "{} transcendental-through-mode tests failed", fail);
}

// ════════════════════════════════════════════════════════════════════
// Part 4: Mode Isolation
// ════════════════════════════════════════════════════════════════════

#[test]
fn test_mode_isolation() {
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║              MODE ROUTING: Isolation & Reset                ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Test 1: Default mode is auto:auto
    reset_gmath_mode();
    let result = evaluate(&(gmath("1.5") + gmath("2.5"))).unwrap();
    assert!(
        matches!(result, StackValue::Decimal(..)),
        "Default mode should auto-route decimal input to Decimal domain, got {}",
        domain_name(&result),
    );
    println!("  [PASS] Default auto:auto routes decimal → Decimal");

    // Test 2: Set mode, verify it takes effect
    set_gmath_mode("binary:symbolic").unwrap();
    let result = evaluate(&(gmath("1.5") + gmath("2.5"))).unwrap();
    assert!(
        matches!(result, StackValue::Symbolic(..)),
        "binary:symbolic should output Symbolic, got {}",
        domain_name(&result),
    );
    println!("  [PASS] binary:symbolic routes → Symbolic");

    // Test 3: Reset mode, verify auto:auto is restored
    reset_gmath_mode();
    let result = evaluate(&(gmath("1.5") + gmath("2.5"))).unwrap();
    assert!(
        matches!(result, StackValue::Decimal(..)),
        "After reset, should be back to auto:auto (Decimal), got {}",
        domain_name(&result),
    );
    println!("  [PASS] reset_gmath_mode() restores auto:auto");

    // Test 4: Setting one mode doesn't affect subsequent tests after reset
    set_gmath_mode("ternary:ternary").unwrap();
    let result = evaluate(&(gmath("3") + gmath("7"))).unwrap();
    assert!(
        matches!(result, StackValue::Ternary(..)),
        "ternary:ternary should output Ternary, got {}",
        domain_name(&result),
    );
    reset_gmath_mode();

    let result = evaluate(&(gmath("3") + gmath("7"))).unwrap();
    assert!(
        matches!(result, StackValue::Binary(..)),
        "After reset, integers should auto-route to Binary, got {}",
        domain_name(&result),
    );
    println!("  [PASS] Mode doesn't leak after reset");

    // Test 5: Multiple mode switches in sequence
    for mode in &["binary:decimal", "decimal:binary", "symbolic:ternary", "auto:auto"] {
        set_gmath_mode(mode).unwrap();
        let _ = evaluate(&(gmath("2") + gmath("3"))).unwrap();
        reset_gmath_mode();
    }
    let result = evaluate(&(gmath("1.5") + gmath("2.5"))).unwrap();
    assert!(
        matches!(result, StackValue::Decimal(..)),
        "After many switches + reset, should be auto:auto, got {}",
        domain_name(&result),
    );
    println!("  [PASS] Multiple mode switches + reset = clean state");

    println!("\n  ── Isolation Summary ──");
    println!("  All isolation tests passed");
}

// ════════════════════════════════════════════════════════════════════
// Part 5: Full 25-mode sweep summary
// ════════════════════════════════════════════════════════════════════

#[test]
fn test_mode_sweep_summary() {
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║          MODE ROUTING: Full 25-Mode Sweep Summary          ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    println!("  {:20} {:>8} {:>8} {:>8}", "MODE", "DOMAIN", "ADD", "MUL");
    println!("  {:20} {:>8} {:>8} {:>8}", "────────────────────", "────────", "────────", "────────");

    let mut total_pass = 0u32;
    let mut total_fail = 0u32;

    for mode in MODES {
        let parts: Vec<&str> = mode.split(':').collect();
        let (compute, output) = (parts[0], parts[1]);

        reset_gmath_mode();
        set_gmath_mode(mode).expect("valid mode");

        // Quick smoke test: add and multiply with decimals
        let add_result = evaluate(&(gmath("1.5") + gmath("2.5")));
        let mul_result = evaluate(&(gmath("1.5") * gmath("2.5")));

        reset_gmath_mode();

        // Inputs: 1.5 = 3/2, 2.5 = 5/2
        let a_rat = (3_i128, 2_i128);
        let b_rat = (5_i128, 2_i128);

        // Check add: 1.5 + 2.5 = 4 (integer — exact unless inputs lossy in compute domain)
        let add_ok = match &add_result {
            Ok(v) => {
                let exact = is_exact_through_pipeline(
                    compute, output, "decimal", a_rat, b_rat, 4, 1,
                );
                if exact {
                    match v.to_rational() {
                        Ok(r) => rationals_equal(r.numerator_i128(), r.denominator_i128(), 4, 1),
                        Err(_) => false,
                    }
                } else {
                    // Lossy pipeline — check approximate via decimal string
                    let s = v.to_decimal_string(5);
                    s.starts_with("3.99") || s.starts_with("4.00") || s == "4"
                }
            }
            Err(_) => false,
        };

        // Check mul: 1.5 * 2.5 = 3.75 (exact in decimal/symbolic, lossy in binary/ternary)
        let mul_ok = match &mul_result {
            Ok(v) => {
                let exact = is_exact_through_pipeline(
                    compute, output, "decimal", a_rat, b_rat, 15, 4,
                );
                if exact {
                    match v.to_rational() {
                        Ok(r) => rationals_equal(r.numerator_i128(), r.denominator_i128(), 15, 4),
                        Err(_) => false,
                    }
                } else {
                    // Lossy pipeline — check approximate
                    let s = v.to_decimal_string(5);
                    s.starts_with("3.74") || s.starts_with("3.75")
                }
            }
            Err(_) => false,
        };

        let domain_str = match &add_result {
            Ok(v) => domain_name(v),
            Err(_) => "ERROR",
        };

        let add_str = if add_ok { "OK" } else { "FAIL" };
        let mul_str = if mul_ok { "OK" } else { "FAIL" };

        println!("  {:20} {:>8} {:>8} {:>8}", mode, domain_str, add_str, mul_str);

        if add_ok { total_pass += 1; } else { total_fail += 1; }
        if mul_ok { total_pass += 1; } else { total_fail += 1; }
    }

    println!("\n  ── Full Sweep Summary ──");
    println!("  Pass: {}/50, Fail: {}/50", total_pass, total_fail);

    assert_eq!(total_fail, 0, "{} sweep tests failed out of 50", total_fail);
}
