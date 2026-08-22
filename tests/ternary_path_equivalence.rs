//! Balanced ternary — 0.4.33 items 5–6: UGOD promotion, canonical/imperative
//! path equivalence, and cross-domain coercion.
//!
//! Completes the suite started in `ternary_domain_validation.rs` (oracle +
//! theorem tests). Here:
//! - UGOD `UniversalTernaryFixed` promotion at raw-overflow boundaries and
//!   tier alignment, with values preserved exactly across promotion.
//! - FASC `0t` literal arithmetic against the imperative UGOD path — the
//!   canonical evaluator dispatches to the same tier functions, so results
//!   must agree exactly.
//! - Cross-domain coercion: a ternary operand mixed with binary, decimal,
//!   and symbolic operands displays identically to the all-plain expression
//!   (router coercion must not change the value).
//! - The `0t` fractional-literal conversion boundary (contract §5) pinned.
//!
//! Profile-independent (UGOD tiers 1–3 are plain integer ops; FASC display
//! comparisons avoid profile-specific raw layouts).

use g_math::canonical::{evaluate, gmath, gmath_parse, LazyExpr, StackValue};
use g_math::fixed_point::domains::balanced_ternary::{
    UniversalTernaryFixed, TernaryTier, SCALE_TQ8_8, SCALE_TQ16_16,
};

// ============================================================================
// Helpers
// ============================================================================

fn ugod(s: &str) -> UniversalTernaryFixed {
    UniversalTernaryFixed::from_str(s).expect("parse")
}

fn display(expr: &LazyExpr) -> String {
    format!("{}", evaluate(expr).expect("evaluate"))
}

/// Runtime-string literal (gmath! requires 'static strs).
fn g(s: &str) -> LazyExpr {
    gmath_parse(s).expect("parse literal")
}

fn is_ternary(v: &StackValue) -> bool {
    matches!(v, StackValue::Ternary(..))
}

// ============================================================================
// Item 5a — UGOD promotion at raw-overflow boundaries
// ============================================================================

#[test]
fn ugod_promotion_on_multiply_overflow() {
    // 3000 × 3000 = 9,000,000 — the value fits easily, but the Tier 1
    // product raw (9e6 · 3^8 ≈ 5.9e13) overflows i32, so UGOD must promote.
    let a = ugod("3000");
    assert_eq!(a.current_tier(), TernaryTier::Tier1);
    let product = a.multiply(&a).expect("multiply with promotion");
    assert_ne!(
        product.current_tier(),
        TernaryTier::Tier1,
        "product must be promoted beyond Tier 1"
    );
    let (tier, raw) = product.to_tier_value();
    assert_eq!(tier, 2, "expected promotion to exactly Tier 2");
    assert_eq!(raw, 9_000_000i128 * SCALE_TQ16_16 as i128, "value preserved exactly");
}

#[test]
fn ugod_promotion_preserves_value_exactly() {
    // Lossless promotion roundtrip: value at Tier 1, promoted through the
    // tiers, must decode to the same rational (raw scales by 3^8 per step).
    let v = ugod("3280"); // the full nominal Tier-1 integer window
    let (t1, raw1) = v.to_tier_value();
    assert_eq!((t1, raw1), (1, 3280 * SCALE_TQ8_8 as i128));
    let promoted = v.promote_to_tier(TernaryTier::Tier2).expect("promote");
    let (t2, raw2) = promoted.to_tier_value();
    assert_eq!(t2, 2);
    assert_eq!(raw2, 3280 * SCALE_TQ16_16 as i128, "promotion must be exact rescale");
}

#[test]
fn ugod_mixed_tier_alignment() {
    // from_integer places small values at Tier 1 and 6-digit values at
    // Tier 1 too (i16-gated) — force a Tier 2 via a large integer.
    let small = UniversalTernaryFixed::from_integer(5).expect("small");
    let large = UniversalTernaryFixed::from_integer(100_000).expect("large");
    assert_eq!(small.current_tier(), TernaryTier::Tier1);
    assert_eq!(large.current_tier(), TernaryTier::Tier2);
    let sum = large.add(&small).expect("mixed-tier add");
    assert_ne!(sum.current_tier(), TernaryTier::Tier1);
    let (tier, raw) = sum.to_tier_value();
    assert_eq!(tier, 2);
    assert_eq!(raw, 100_005i128 * SCALE_TQ16_16 as i128, "aligned add exact");
}

#[test]
fn ugod_window_boundaries_exact() {
    // ±3280 = ±(3^8−1)/2 — the all-(+1)/(−1) integer-trit patterns of Tier 1.
    for (s, expected) in [("3280", 3280i128), ("1", 1), ("0", 0)] {
        let v = ugod(s);
        let (_t, raw) = v.to_tier_value();
        assert_eq!(raw, expected * SCALE_TQ8_8 as i128, "window value {s}");
    }
    // from_str window gate: 3281 exceeds the Tier-1 window and must land
    // on Tier 2 (contract §1b range note: from_str DOES enforce the nominal
    // window, unlike from_integer).
    let over = ugod("3281");
    assert_eq!(over.current_tier(), TernaryTier::Tier2);
    let (_, raw) = over.to_tier_value();
    assert_eq!(raw, 3281 * SCALE_TQ16_16 as i128);
}

// ============================================================================
// Item 5b — canonical (FASC) vs imperative (UGOD) path equivalence
// ============================================================================

#[test]
fn fasc_ternary_arithmetic_stays_ternary_and_exact() {
    // Integer 0t literals: results must be exact and remain in the ternary
    // domain (no silent coercion), and agree with direct UGOD arithmetic.
    let cases: &[(&str, &str, &str, &str)] = &[
        // (op, a, b, expected-integer)
        ("+", "7", "5", "12"),
        ("+", "3280", "1", "3281"),
        ("-", "7", "12", "-5"),
        ("*", "21", "3", "63"),
        ("*", "81", "81", "6561"),
        ("/", "63", "3", "21"),
    ];
    for &(op, a, b, expected) in cases {
        let (ea, eb) = (g(&format!("0t{a}")), g(&format!("0t{b}")));
        let expr = match op {
            "+" => ea + eb,
            "-" => ea - eb,
            "*" => ea * eb,
            "/" => ea / eb,
            _ => unreachable!(),
        };
        let result = evaluate(&expr).expect("fasc evaluate");
        assert!(is_ternary(&result), "0t{a} {op} 0t{b} left the ternary domain");
        // Ternary integer results display as plain integers on all profiles.
        assert_eq!(
            format!("{result}"),
            expected,
            "FASC 0t{a} {op} 0t{b} != {expected}"
        );
        // And equals the imperative UGOD computation, integer-exactly.
        let ua = ugod(a);
        let ub = ugod(b);
        let ures = match op {
            "+" => ua.add(&ub),
            "-" => ua.subtract(&ub),
            "*" => ua.multiply(&ub),
            "/" => ua.divide(&ub),
            _ => unreachable!(),
        }
        .expect("ugod op");
        let (tier, raw) = ures.to_tier_value();
        let scale: i128 = match tier {
            1 => SCALE_TQ8_8 as i128,
            2 => SCALE_TQ16_16 as i128,
            t => panic!("unexpected tier {t} for small integers"),
        };
        assert_eq!(raw % scale, 0, "integer result must be exact");
        assert_eq!((raw / scale).to_string(), expected, "UGOD path diverges");
    }
}

#[test]
fn fasc_ternary_negation_symmetry() {
    // NOTE: a leading-minus literal ("-0t7") is a ParseError — the 0t prefix
    // check requires position 0. Negation is expressed as an operator.
    for s in ["1", "7", "3280"] {
        let pos = display(&g(&format!("0t{s}")));
        let negated = display(&(-g(&format!("0t{s}"))));
        let via_sub = display(&(g("0t0") - g(&format!("0t{s}"))));
        assert_eq!(negated, via_sub, "negation path divergence at 0t{s}");
        assert_ne!(pos, negated);
        // And double negation restores the original display.
        let double = display(&(-(-g(&format!("0t{s}")))));
        assert_eq!(double, pos, "neg(neg(0t{s})) != 0t{s}");
    }
}

// ============================================================================
// Item 5c — cross-domain coercion must not change values
// ============================================================================

#[test]
fn cross_domain_coercion_matches_plain_expressions() {
    // Each pair: expression with a ternary operand vs the same expression
    // with a plain operand. Router coercion must produce identical display.
    let pairs: &[(fn() -> LazyExpr, fn() -> LazyExpr)] = &[
        (|| gmath("0t2") + gmath("3"), || gmath("2") + gmath("3")),
        (|| gmath("0t2") + gmath("0.5"), || gmath("2") + gmath("0.5")),
        (|| gmath("0t2") * gmath("1/3"), || gmath("2") * gmath("1/3")),
        (|| gmath("0t7") - gmath("0.25"), || gmath("7") - gmath("0.25")),
        (|| gmath("10") / gmath("0t4"), || gmath("10") / gmath("4")),
    ];
    // Display formatting differs per result domain (ternary "5" vs decimal
    // "5.000000000"), so equality is checked through the router itself:
    // (ternary_expr − plain_expr) must evaluate to exactly zero.
    for (i, (ternary_side, plain_side)) in pairs.iter().enumerate() {
        let diff = display(&(ternary_side() - plain_side()));
        let trimmed = if diff.contains('.') {
            diff.trim_end_matches('0').trim_end_matches('.')
        } else {
            diff.as_str()
        };
        assert!(
            trimmed == "0" || trimmed == "-0" || trimmed == "0/1",
            "cross-domain pair {i} diverges: difference displays as {diff:?}"
        );
    }
}

// ============================================================================
// Contract §5 — the 0t fractional-literal conversion boundary
// ============================================================================

#[test]
fn fractional_literal_conversion_boundary() {
    // "0t0.5" forces the canonical tie: 0.5·3^8 = 3280.5, exactly between
    // grid points (3^F odd — contract §4/§5). Pin the shipped conversion.
    let v = UniversalTernaryFixed::from_str("0.5").expect("parse 0.5");
    let (tier, raw) = v.to_tier_value();
    assert_eq!(tier, 1);
    // MEASURED pin (contract §5): the shipped conversion truncates toward
    // zero — 0.5·3^8 = 3280.5 lands on 3280 (the tie resolves low), and
    // 1.5·3^8 = 9841.5 on 9841.
    assert_eq!(raw, 3280, "0.5 conversion pin");
    let (_, raw15) = UniversalTernaryFixed::from_str("1.5").expect("1.5").to_tier_value();
    assert_eq!(raw15, 9841, "1.5 conversion pin");
    // Exact-decimal values that ARE ternary-representable: integers only —
    // 0.5 is necessarily inexact (denominator 2 ∉ {3}).
    assert_ne!(raw as i128 * 2, (SCALE_TQ8_8 as i128) * 1, "0.5 cannot be exact");
}

#[test]
fn ternary_literal_tier2_storage_limit_is_loud() {
    // from_str window-gates 3281 to Tier 2 (raw 3281·3^16 ≈ 1.4e11). On the
    // narrow profiles that raw does not fit BinaryStorage — before 0.4.33
    // this silently wrapped into garbage (0t3281 displayed as "-11.56021"
    // on realtime); now it is a loud TierOverflow at parse.
    // Note the asymmetry: the same VALUE reached by arithmetic
    // (0t3280 + 0t1) stays at Tier 1 raw 3281·3^8, which fits everywhere.
    let parsed = gmath_parse("0t3281");
    #[cfg(any(table_format = "q16_16"))]
    assert!(parsed.is_err(), "Tier-2 literal must fail loud on realtime");
    #[cfg(not(any(table_format = "q16_16")))]
    {
        let expr = parsed.expect("wider profiles hold Tier-2 raws");
        assert_eq!(display(&expr), "3281");
    }
}

#[test]
fn negative_fractional_literal_sign_regression() {
    // REGRESSION (fixed in 0.4.33): "-0.5" parsed as +0.5 — the "-0"
    // integer part lost its sign before the fraction was applied. The fix
    // strips the sign once in from_str and negates the parsed magnitude.
    for (s, expected_raw) in [("-0.5", -3280i128), ("-0.1", -656), ("-1.5", -9841)] {
        let (tier, raw) = UniversalTernaryFixed::from_str(s)
            .expect("parse negative fractional")
            .to_tier_value();
        assert_eq!(tier, 1);
        assert_eq!(raw, expected_raw, "sign-correct conversion of {s}");
    }
    // Sign symmetry of conversion: parse(-s) == -parse(s), exactly.
    for s in ["0.5", "0.1", "1.5", "3280", "0.0001"] {
        let pos = UniversalTernaryFixed::from_str(s).expect("pos").to_tier_value();
        let neg = UniversalTernaryFixed::from_str(&format!("-{s}")).expect("neg").to_tier_value();
        assert_eq!(neg.0, pos.0, "tier symmetry for {s}");
        assert_eq!(neg.1, -pos.1, "raw symmetry for {s}");
    }
    // Double minus stays rejected.
    assert!(UniversalTernaryFixed::from_str("--5").is_err());
}

// ============================================================================
// Gap-closing (post-0.4.33): wide ternary_to_storage arms + FASC
// transcendentals on ternary operands
// ============================================================================

#[test]
fn ternary_literals_cap_at_tier3() {
    // from_str parses the integer part as i64 (≤ ~9.2e18), and Tier 3 has
    // no window gate — any i64 integer's raw (≤ 9.2e18·3^32 ≈ 1.7e34) fits
    // i128. Consequence, pinned here: 0t literals can NEVER land beyond
    // Tier 3; the Medium/Large/XLarge storage arms are reachable only via
    // internal from_tier_raw paths (unit-tested in domain.rs).
    let v = UniversalTernaryFixed::from_str("9000000000000000000").expect("i64-max-ish");
    assert_eq!(v.current_tier(), TernaryTier::Tier3);
    // And the same literal through FASC: loud error on narrow storage,
    // exact on i128+ storage.
    let parsed = gmath_parse("0t9000000000000000000");
    #[cfg(any(table_format = "q16_16", table_format = "q32_32"))]
    assert!(parsed.is_err(), "big Tier-3 literal must fail loud on narrow storage");
    #[cfg(not(any(table_format = "q16_16", table_format = "q32_32")))]
    {
        let expr = parsed.expect("i128+ storage holds Tier-3 raws");
        assert_eq!(display(&expr), "9000000000000000000");
    }
}

#[test]
fn tier3_literal_storage_conversion_per_profile() {
    // 9e14 is inside the Tier-3 window: raw = 9e14 · 3^32 ≈ 1.67e30 —
    // fits i128 (embedded+) but not i64/i32 (realtime, compact).
    let parsed = gmath_parse("0t900000000000000");
    #[cfg(any(table_format = "q16_16", table_format = "q32_32"))]
    assert!(parsed.is_err(), "Tier-3 literal must fail loud on realtime/compact");
    #[cfg(not(any(table_format = "q16_16", table_format = "q32_32")))]
    {
        let expr = parsed.expect("i128+ storage holds Tier-3 raws");
        assert_eq!(display(&expr), "900000000000000");
    }
}

#[test]
fn fasc_transcendentals_on_ternary_operands_match_plain() {
    // Ternary operands route through the binary compute-tier engines; the
    // result must be identical to the same computation on a plain literal
    // (path independence across the ternary entry point).
    let cases: &[(&str, &str)] = &[("0t2", "2"), ("0t9", "9"), ("0t3280", "3280")];
    for &(t, p) in cases {
        assert_eq!(
            display(&g(t).sqrt()),
            display(&g(p).sqrt()),
            "sqrt diverges for ternary operand {t}"
        );
        assert_eq!(
            display(&g(t).ln()),
            display(&g(p).ln()),
            "ln diverges for ternary operand {t}"
        );
    }
    // exp/sin on a small ternary operand, plus a composed chain.
    assert_eq!(display(&g("0t2").exp()), display(&g("2").exp()));
    assert_eq!(display(&g("0t2").sin()), display(&g("2").sin()));
    // Composed chain: the ternary-origin result may materialize back into
    // the ternary domain ("2") while the plain path stays binary-formatted
    // ("2.0000000000000000000") — same VALUE, different display. Compare
    // through the router: the difference must be exactly zero.
    let diff = display(&(g("0t2").exp().ln() - g("2").exp().ln()));
    let trimmed = if diff.contains('.') {
        diff.trim_end_matches('0').trim_end_matches('.')
    } else {
        diff.as_str()
    };
    assert!(
        trimmed == "0" || trimmed == "-0",
        "composed chain diverges for ternary operand: diff = {diff:?}"
    );
}
