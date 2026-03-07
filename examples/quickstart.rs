//! gMath Quickstart — Zero-float fixed-point arithmetic
//!
//! Run: GMATH_PROFILE=embedded cargo run --example quickstart

use g_math::canonical::{gmath, gmath_parse, evaluate, LazyExpr};

fn main() {
    // ── Basic arithmetic ────────────────────────────────────────────────
    // gmath() builds lazy expression trees — no computation until evaluate()
    let sum = gmath("1.5") + gmath("2.5");
    let product = gmath("7") * gmath("6");
    let compound = (gmath("100") + gmath("50")) / gmath("3");

    println!("1.5 + 2.5  = {}", evaluate(&sum).unwrap());
    println!("7 * 6      = {}", evaluate(&product).unwrap());
    println!("150 / 3    = {}", evaluate(&compound).unwrap());

    // ── Transcendental functions (18 total, tier N+1 computation) ───────
    let e       = evaluate(&gmath("1").exp()).unwrap();       // e^1
    let ln2     = evaluate(&gmath("2").ln()).unwrap();        // ln(2)
    let sqrt2   = evaluate(&gmath("2").sqrt()).unwrap();      // sqrt(2)
    let sin_1   = evaluate(&gmath("1").sin()).unwrap();       // sin(1)
    let cos_1   = evaluate(&gmath("1").cos()).unwrap();       // cos(1)
    let atan_1  = evaluate(&gmath("1").atan()).unwrap();      // atan(1) = pi/4

    println!("\ne          = {}", e);
    println!("ln(2)      = {}", ln2);
    println!("sqrt(2)    = {}", sqrt2);
    println!("sin(1)     = {}", sin_1);
    println!("cos(1)     = {}", cos_1);
    println!("atan(1)    = {}", atan_1);

    // ── Chaining expressions ────────────────────────────────────────────
    // Compose transcendentals — BinaryCompute keeps full precision through the chain
    let chain = gmath("2").ln().exp();  // exp(ln(2)) = 2
    println!("\nexp(ln(2)) = {}", evaluate(&chain).unwrap());

    let sin_cos = gmath("0.5").sin().pow(gmath("2")) + gmath("0.5").cos().pow(gmath("2"));
    println!("sin^2 + cos^2 = {}", evaluate(&sin_cos).unwrap());  // = 1

    // ── Runtime string parsing ──────────────────────────────────────────
    // gmath_parse() accepts runtime &str — same precision, eager parse
    let user_input = "3.14159265358979323846";
    let parsed = gmath_parse(user_input).unwrap();
    let result = evaluate(&(parsed * gmath("2"))).unwrap();
    println!("\n2 * pi     = {}", result);

    // ── Feeding results back ────────────────────────────────────────────
    // LazyExpr::from(StackValue) preserves full precision + shadow
    let balance = evaluate(&gmath("1000")).unwrap();
    let rate = gmath("1.05");
    let year1 = evaluate(&(LazyExpr::from(balance) * rate)).unwrap();
    let year2 = evaluate(&(LazyExpr::from(year1.clone()) * gmath("1.05"))).unwrap();
    println!("\nCompound interest:");
    println!("  Year 0: 1000");
    println!("  Year 1: {}", year1);
    println!("  Year 2: {}", year2);

    // ── Symbolic/rational domain ────────────────────────────────────────
    // Repeating decimals and fractions route to exact symbolic arithmetic
    let third = evaluate(&gmath("1/3")).unwrap();
    let sixth = evaluate(&(gmath("1/3") + gmath("1/6"))).unwrap();
    println!("\n1/3        = {}", third);
    println!("1/3 + 1/6  = {}", sixth);  // exact: 1/2
}
