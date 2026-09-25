//! 0.5.0 item 1 — UGOD multi-tier promotion verification (ROADMAP).
//!
//! Verdict encoded here: the mid-ladder (tiers 1→4) always promoted
//! correctly; the TOP of every ladder was broken and is now fixed:
//! - binary Tier4/5 mul truncated I512→I256 unchecked (1e20 × 1e20 on
//!   balanced returned wrapped garbage), no 4→5→6 promotion arms
//! - binary Tier4-6 add/sub used bare wrapping operators
//! - binary_to_storage narrowed promoted raws with bare casts (9e18+9e18
//!   on embedded returned 0.0) — binary twin of the 0.4.33 ternary fix
//! - divide mislabeled quotient overflow as DivisionByZero and stopped
//! - the SYMBOLIC ladder's "promotion" retried i128×i128 at the same
//!   width — Huge×Huge never reached the existing Massive (I256) tier
//! - FASC binary arms now fall back to the exact rational path when the
//!   promoted result exceeds profile storage: the true ladder top.
//!
//! Contract these tests pin: arithmetic on representable inputs either
//! returns the EXACT value (possibly via a wider tier or the symbolic
//! domain) or fails LOUD — never a silent wrap.

use g_math::canonical::{evaluate, gmath, LazyExpr};

fn disp(e: &LazyExpr) -> String {
    format!("{}", evaluate(e).expect("must evaluate"))
}

fn int_part(s: &str) -> String {
    s.split('.').next().unwrap().to_string()
}

#[test]
fn promoted_add_is_exact_never_wraps() {
    // Pre-fix on embedded: promoted Tier-4 sum wrapped through as_i128 to 0.0.
    assert_eq!(
        int_part(&disp(&(gmath("9000000000000000000") + gmath("9000000000000000000")))),
        "18000000000000000000"
    );
    assert_eq!(
        int_part(&disp(&(-gmath("9000000000000000000") - gmath("9000000000000000000")))),
        "-18000000000000000000"
    );
}

#[test]
fn promoted_multiply_is_exact_never_wraps() {
    // Pre-fix on balanced: returned 1.318e38 (silent I512→I256 truncation).
    // Pre-fix everywhere: the symbolic ladder couldn't hold 1e40 either.
    let want = format!("1{}", "0".repeat(40));
    assert_eq!(
        int_part(&disp(&(gmath("100000000000000000000") * gmath("100000000000000000000")))),
        want
    );
    // Negative operand keeps exactness through the same ladder.
    let got = int_part(&disp(&(-gmath("100000000000000000000") * gmath("100000000000000000000"))));
    assert_eq!(got, format!("-{want}"));
}

#[test]
fn promoted_divide_is_exact_or_loud() {
    // Tiny reciprocal: 1e-20 is below one ulp on narrow profiles (Q64.64
    // ulp ≈ 5.4e-20), so its DISPLAY is profile-dependent — but the value
    // must stay EXACT through the chain: (1/1e20)·1e20 = 1 precisely.
    // Pre-fix, the unchecked symbolic→binary coercion wrapped 1e20 mod
    // 2^128 and this round-trip returned garbage.
    let r = disp(&((gmath("1") / gmath("100000000000000000000"))
        * gmath("100000000000000000000")));
    assert_eq!(int_part(&r), "1", "1e-20 round-trip lost exactness: {r}");
    // Huge quotient escapes binary storage → rational fallback, exact.
    assert_eq!(
        int_part(&disp(&(gmath("9000000000000000000") / gmath("0.000000001")))),
        "9000000000000000000000000000"
    );
}

#[test]
fn division_by_zero_is_labeled_correctly() {
    // Pre-fix, a Tier-4 quotient OVERFLOW was mislabeled DivisionByZero.
    // Now: zero divisor → DivisionByZero (decided once at ladder entry);
    // overflow → exact rational fallback (no error at all).
    use g_math::canonical::gmath_parse;
    let e = gmath_parse("1").unwrap() / gmath_parse("0").unwrap();
    match evaluate(&e) {
        Err(err) => assert!(
            format!("{err:?}").contains("DivisionByZero"),
            "wrong error kind: {err:?}"
        ),
        Ok(v) => panic!("division by zero must not succeed, got {v}"),
    }
}

#[test]
fn symbolic_ladder_reaches_wide_tiers() {
    // A fraction literal is Symbolic on EVERY profile (a plain 1e20/3
    // divide is native binary on scientific, where 1e20 fits storage,
    // and would honestly round instead). The product needs an I256
    // numerator: the symbolic ladder must reach the Massive tier.
    let got = disp(&(gmath("100000000000000000000/3")
        * gmath("300000000000000000000")));
    // (1e20/3)·3e20 = 1e40 exactly.
    assert_eq!(int_part(&got), format!("1{}", "0".repeat(40)));
}

// ============================================================================
// 0.6.4: canonical literals and decimal results never narrow by wrapping
// ============================================================================

fn is_zero(e: &LazyExpr) -> bool {
    disp(e).trim_start_matches('-').chars().all(|c| c == '0' || c == '.')
}

#[test]
fn decimal_literals_beyond_the_decimal_storage_keep_their_value() {
    // Realtime keeps value * 10^4 in i32 decimal storage. 0.6.3 wrapped it:
    // at Q22.10 "467295.6470" was 37798.9 and "214748.3648" was -214748.3648.
    assert!(is_zero(&(gmath("467295.6470") - gmath("467295.647"))), "{}", disp(&gmath("467295.6470")));
    assert!(is_zero(&(gmath("214748.3648") - gmath("2147483648/10000"))), "{}", disp(&gmath("214748.3648")));
    assert!(is_zero(&(gmath("-214748.3649") + gmath("2147483649/10000"))), "{}", disp(&gmath("-214748.3649")));
    assert!(disp(&gmath("3000000.0001")).starts_with("3000000.0001"), "{}", disp(&gmath("3000000.0001")));
}

#[test]
fn decimal_results_beyond_the_decimal_storage_are_exact() {
    // 0.6.3 narrowed every decimal UGOD result back to storage with a
    // truncating cast: at Q22.10, 214748.3647 + 1 = -214747.3649.
    assert!(is_zero(&(gmath("214748.3647") + gmath("1") - gmath("2147493647/10000"))),
        "{}", disp(&(gmath("214748.3647") + gmath("1"))));
    assert!(is_zero(&(gmath("214748.3647") + gmath("0.0001") - gmath("2147483648/10000"))));
    assert!(is_zero(&(gmath("-214748.3647") - gmath("0.0002") + gmath("2147483649/10000"))));
    assert!(is_zero(&(gmath("107374.1824") * gmath("2.0000") - gmath("2147483648/10000"))));
}

#[test]
fn hex_and_binary_literals_denote_their_integer() {
    // 0.6.3 stored the digits as raw storage bits at tier 1 while the shadow
    // said 255/1: "0xFF" alone, "0xFF" + 1 and the shadow disagreed, and
    // "0x1FFFFFFFF" wrapped to 0 on realtime.
    assert!(is_zero(&(gmath("0xFF") - gmath("255"))), "{}", disp(&gmath("0xFF")));
    assert!(is_zero(&(gmath("0xFF") + gmath("1") - gmath("256"))));
    assert!(is_zero(&(gmath("0b101") - gmath("5"))));
    assert!(is_zero(&(gmath("0x1FFFFFFFF") - gmath("8589934591"))), "{}", disp(&gmath("0x1FFFFFFFF")));
}

#[test]
fn canonical_literal_grammar_is_unchanged() {
    // The exact converter behind the 0.6.4 fixes reads exponent notation;
    // the canonical grammar must not (FixedPoint::from_str does).
    use g_math::canonical::gmath_parse;
    for s in ["1e5", "1.5e3", "2E-3", " 1.5", "1.5 "] {
        assert!(gmath_parse(s).is_err(), "{s:?} parsed");
    }
}
