//! Balanced ternary domain validation — 0.4.33 items 2–4.
//!
//! Implements the invariant checklist of
//! `docs/design/BALANCED_TERNARY_CONTRACT.md` §7:
//! a deliberately boring trit-vector reference oracle, exhaustive
//! small-range equivalence against the shipping tier arithmetic, boundary
//! families, and the tie-free rounding theorem with its conversion-boundary
//! caveat.
//!
//! The oracle here is a *native trit* implementation (signed digit vectors,
//! trit-wise carry propagation, shift-and-add multiply). The UGOD tiers it
//! is checked against are *scaled integers* (contract §1a vs §1b), so these
//! tests double as the bridge proving the two representations agree —
//! `theorem_trit_truncation_is_round_nearest` most directly. The native-trit
//! operations that ship (packing, zero-multiply dots) have their own suite
//! in `tq19_validation.rs`.
//!
//! Profile-independent: Tier 1/2 ternary functions are plain integer ops
//! with no `table_format` gating. Runs under every profile and feature set.

use g_math::fixed_point::domains::balanced_ternary::{
    add_ternary_tq10_10, divide_ternary_tq10_10, multiply_ternary_tq10_10,
    negate_ternary_tq10_10, subtract_ternary_tq10_10, SCALE_TQ10_10,
};

// ============================================================================
// Reference oracle — signed trit vectors, little-endian, each exactly -1/0/+1.
// Deliberately slow and obvious; no clever encodings.
// ============================================================================

/// Canonical balanced-ternary digits of `n` (little-endian). Empty for 0.
fn encode(mut n: i128) -> Vec<i8> {
    let mut trits = Vec::new();
    while n != 0 {
        // Balanced digit in {-1, 0, +1} from the euclidean remainder.
        let d: i8 = match n.rem_euclid(3) {
            0 => 0,
            1 => 1,
            2 => -1,
            _ => unreachable!(),
        };
        trits.push(d);
        n = (n - d as i128) / 3;
    }
    trits
}

fn decode(trits: &[i8]) -> i128 {
    let mut value = 0i128;
    for &d in trits.iter().rev() {
        value = value * 3 + d as i128;
    }
    value
}

/// Trit-wise addition with carry propagation. Exact.
fn ref_add(a: &[i8], b: &[i8]) -> Vec<i8> {
    let len = a.len().max(b.len()) + 2;
    let mut out = Vec::with_capacity(len);
    let mut carry = 0i8;
    for i in 0..len {
        let s = carry
            + a.get(i).copied().unwrap_or(0)
            + b.get(i).copied().unwrap_or(0);
        // s in [-3, 3]; balanced digit + new carry.
        let (d, c) = match s {
            -3 => (0, -1),
            -2 => (1, -1),
            -1 => (-1, 0),
            0 => (0, 0),
            1 => (1, 0),
            2 => (-1, 1),
            3 => (0, 1),
            _ => unreachable!(),
        };
        out.push(d);
        carry = c;
    }
    assert_eq!(carry, 0, "oracle add carry must resolve within padded length");
    while out.last() == Some(&0) {
        out.pop();
    }
    out
}

/// Negation = trit-wise flip. Exact, total, no overflow case exists.
fn ref_neg(a: &[i8]) -> Vec<i8> {
    a.iter().map(|&d| -d).collect()
}

/// Shift-and-add multiplication in trit space. Exact.
fn ref_mul(a: &[i8], b: &[i8]) -> Vec<i8> {
    let mut acc = Vec::new();
    for (i, &d) in b.iter().enumerate() {
        if d == 0 {
            continue;
        }
        // partial = a * d * 3^i  (d is ±1, shift = i leading zeros)
        let mut partial = vec![0i8; i];
        partial.extend(a.iter().map(|&t| t * d));
        acc = ref_add(&acc, &partial);
    }
    acc
}

/// Nearest multiple of 3^m via balanced remainder (the theorem's rounding).
/// Returns (rounded_value, balanced_remainder).
fn ref_round_nearest(n: i128, m: u32) -> (i128, i128) {
    let p = 3i128.pow(m);
    let mut r = n.rem_euclid(p); // in [0, p)
    if r > (p - 1) / 2 {
        r -= p; // balanced: in [-(p-1)/2, +(p-1)/2]
    }
    (n - r, r)
}

// ============================================================================
// Oracle self-consistency + equivalence with shipping arithmetic
// ============================================================================

const EXHAUSTIVE: i128 = 400; // full cross-product range for add/mul oracles
const SWEEP: i128 = 100_000; // strided single-value sweep
const STRIDE: i128 = 617; // prime stride, coprime to 3

#[test]
fn oracle_roundtrip_exhaustive() {
    for n in -SWEEP..=SWEEP {
        assert_eq!(decode(&encode(n)), n, "roundtrip failed at {n}");
    }
    for k in 0..38 {
        let p = 3i128.pow(k);
        for n in [p, -p, p + 1, p - 1, -p - 1, -p + 1] {
            assert_eq!(decode(&encode(n)), n, "roundtrip failed at {n}");
        }
    }
    assert_eq!(decode(&encode(i32::MAX as i128)), i32::MAX as i128);
    assert_eq!(decode(&encode(i32::MIN as i128)), i32::MIN as i128);
}

#[test]
fn oracle_digits_balanced() {
    for n in (-SWEEP..=SWEEP).step_by(7) {
        for &d in &encode(n) {
            assert!((-1..=1).contains(&d), "non-balanced digit {d} for {n}");
        }
        let e = encode(n);
        assert_ne!(e.last(), Some(&0), "non-canonical leading zero for {n}");
    }
}

#[test]
fn oracle_add_matches_raw_add() {
    // Exhaustive cross-product on a small window.
    for a in -EXHAUSTIVE..=EXHAUSTIVE {
        for b in -EXHAUSTIVE..=EXHAUSTIVE {
            let via_oracle = decode(&ref_add(&encode(a), &encode(b)));
            assert_eq!(via_oracle, a + b, "oracle add wrong at {a}+{b}");
        }
    }
    // The shipping tier add is exact integer add on raws; equate all three.
    let mut a = -SWEEP;
    while a <= SWEEP {
        let b = a.wrapping_mul(31) % SWEEP;
        let got = add_ternary_tq10_10(a as i32, b as i32).unwrap();
        assert_eq!(got as i128, decode(&ref_add(&encode(a), &encode(b))));
        let got_sub = subtract_ternary_tq10_10(a as i32, b as i32).unwrap();
        assert_eq!(
            got_sub as i128,
            decode(&ref_add(&encode(a), &ref_neg(&encode(b))))
        );
        a += STRIDE;
    }
}

#[test]
fn oracle_neg_matches_raw_neg() {
    let mut n = -SWEEP;
    while n <= SWEEP {
        let via_oracle = decode(&ref_neg(&encode(n)));
        assert_eq!(via_oracle, -n, "oracle neg wrong at {n}");
        let got = negate_ternary_tq10_10(n as i32).unwrap();
        assert_eq!(got as i128, -n);
        n += STRIDE;
    }
}

#[test]
fn negation_involution_and_inverse() {
    let mut n = -SWEEP;
    while n <= SWEEP {
        let e = encode(n);
        assert_eq!(decode(&ref_neg(&ref_neg(&e))), n, "neg(neg({n})) != {n}");
        assert_eq!(decode(&ref_add(&e, &ref_neg(&e))), 0, "{n} + (-{n}) != 0");
        let raw = negate_ternary_tq10_10(negate_ternary_tq10_10(n as i32).unwrap()).unwrap();
        assert_eq!(raw as i128, n);
        n += STRIDE;
    }
}

#[test]
fn oracle_mul_matches_exact_product() {
    const MUL_RANGE: i128 = 200; // trit-space shift-and-add is the slow path
    for a in -MUL_RANGE..=MUL_RANGE {
        for b in -MUL_RANGE..=MUL_RANGE {
            let via_oracle = decode(&ref_mul(&encode(a), &encode(b)));
            assert_eq!(via_oracle, a * b, "oracle mul wrong at {a}*{b}");
        }
    }
}

// ============================================================================
// Shipping mul/div semantics: toward-zero truncation, odd symmetry
// ============================================================================

/// Round-to-nearest model for p/q: mul uses the symmetric form (odd scale,
/// tie-free); div uses ties toward +∞ (arbitrary divisor CAN tie) —
/// contract §3 as unified in 0.5.0.
fn nearest_symmetric(p: i128, q: i128) -> i128 {
    let t = p / q;
    let r = p - t * q;
    if 2 * r.abs() > q.abs() { t + r.signum() * q.signum() } else { t }
}
fn nearest_ties_up(p: i128, q: i128) -> i128 {
    let t = p / q;
    let r2 = 2 * (p - t * q).abs();
    let qa = q.abs();
    let positive = (p < 0) == (q < 0);
    if if positive { r2 >= qa } else { r2 > qa } {
        t + if positive { 1 } else { -1 }
    } else {
        t
    }
}

#[test]
fn mul_div_nearest_and_symmetric() {
    let scale = SCALE_TQ10_10 as i128;
    let mut a = -SWEEP;
    while a <= SWEEP {
        let mut b = -SWEEP;
        while b <= SWEEP {
            // Contract §3 (0.5.0): mul = nearest((a·b)/3^8) — tie-free
            // because the scale is odd, hence fully sign-symmetric.
            let want = nearest_symmetric(a * b, scale);
            let got = multiply_ternary_tq10_10(a as i32, b as i32).unwrap();
            assert_eq!(got as i128, want, "mul semantics at {a},{b}");
            let got_neg = multiply_ternary_tq10_10(-a as i32, b as i32).unwrap();
            assert_eq!(got_neg, -got, "mul odd-symmetry at {a},{b}");
            if b != 0 {
                // div = nearest((a·3^8)/b), ties toward +∞.
                let want_div = nearest_ties_up(a * scale, b);
                let got_div = divide_ternary_tq10_10(a as i32, b as i32).unwrap();
                assert_eq!(got_div as i128, want_div, "div semantics at {a},{b}");
                // Div symmetry holds except on exact ties (ties-+∞ is not
                // odd-symmetric); the model equality above covers both signs.
                let got_div_neg = divide_ternary_tq10_10(-a as i32, b as i32).unwrap();
                assert_eq!(
                    got_div_neg as i128,
                    nearest_ties_up(-a * scale, b),
                    "div model at -{a},{b}"
                );
            }
            b += STRIDE * 3 + 1; // stride not a multiple of 3
        }
        a += STRIDE;
    }
}

// ============================================================================
// The tie-free rounding theorem (contract §4)
// ============================================================================

#[test]
fn theorem_no_ties_balanced_remainder() {
    for m in 1..=8u32 {
        let p = 3i128.pow(m);
        let mut n = -SWEEP;
        while n <= SWEEP {
            let (rounded, r) = ref_round_nearest(n, m);
            assert_eq!(rounded + r, n);
            assert_eq!(rounded.rem_euclid(p), 0, "not a grid multiple at {n}, m={m}");
            // Strict half-ulp bound: |r| <= (p-1)/2 < p/2. Since p is odd,
            // 2|r| == p is impossible — the tie case does not exist.
            assert!(2 * r.abs() < p, "tie or over-half remainder at {n}, m={m}");
            assert!(r.abs() <= (p - 1) / 2);
            // Uniqueness of the nearest multiple: both neighbors are farther.
            assert!((n - (rounded + p)).abs() > r.abs());
            assert!((n - (rounded - p)).abs() > r.abs());
            n += 41; // dense-ish sweep, coprime to 3
        }
    }
}

#[test]
fn theorem_trit_truncation_is_round_nearest() {
    for m in 1..=6u32 {
        let mut n = -SWEEP;
        while n <= SWEEP {
            // Drop the lowest m trits of the canonical expansion...
            let mut e = encode(n);
            let keep = e.split_off((m as usize).min(e.len()));
            let dropped_tail = decode(&e);
            let mut kept_shifted = vec![0i8; m as usize];
            kept_shifted.extend(keep);
            let truncated = decode(&kept_shifted);
            // ...and it equals the balanced-remainder nearest multiple.
            let (rounded, r) = ref_round_nearest(n, m);
            assert_eq!(truncated, rounded, "trit-drop != nearest at {n}, m={m}");
            assert_eq!(dropped_tail, r, "dropped tail != balanced remainder");
            n += 37;
        }
    }
}

#[test]
fn theorem_rounding_symmetry() {
    for m in 1..=8u32 {
        let mut n = -SWEEP;
        while n <= SWEEP {
            let (r_pos, _) = ref_round_nearest(n, m);
            let (r_neg, _) = ref_round_nearest(-n, m);
            assert_eq!(r_neg, -r_pos, "round(-n) != -round(n) at {n}, m={m}");
            n += 53;
        }
    }
}

// ============================================================================
// Boundary rules (contract §5) and boundary families
// ============================================================================

#[test]
fn boundary_half_is_exact_tie() {
    // Every scale 3^F is odd, so ½·3^F is never an integer: ½ is not
    // representable, and its two neighbors are exactly equidistant — the
    // genuine tie lives at the conversion boundary, not inside the domain.
    let scale = SCALE_TQ10_10 as i128; // 3^8 = 6561, odd
    assert_eq!(scale % 2, 1);
    let below = (scale - 1) / 2; // 3280  -> 3280/6561 < 1/2
    let above = below + 1; //         3281  -> 3281/6561 > 1/2
    // Exact equidistance in doubled integers: |2·below − scale| == |2·above − scale| == 1.
    assert_eq!(scale - 2 * below, 1);
    assert_eq!(2 * above - scale, 1);
}

#[test]
fn boundary_families() {
    let mut cases: Vec<i128> = Vec::new();
    for k in 0..=19u32 {
        let p = 3i128.pow(k);
        cases.extend([p, -p, p + 1, p - 1, -(p + 1), -(p - 1)]);
    }
    // Runs of +1 trits: (3^m − 1)/2; runs of −1 trits: negation.
    for m in 1..=19u32 {
        let run = (3i128.pow(m) - 1) / 2;
        cases.extend([run, -run]);
        // Alternating +1/−1 pattern of length m (little-endian d_i = (−1)^i).
        let alt = decode(
            &(0..m as usize)
                .map(|i| if i % 2 == 0 { 1i8 } else { -1i8 })
                .collect::<Vec<_>>(),
        );
        cases.extend([alt, -alt]);
    }
    for &n in &cases {
        assert_eq!(decode(&encode(n)), n, "roundtrip at boundary {n}");
        assert_eq!(decode(&ref_neg(&encode(n))), -n, "neg at boundary {n}");
        for &other in &[1i128, -1, 3, -3] {
            let sum = decode(&ref_add(&encode(n), &encode(other)));
            assert_eq!(sum, n + other, "add at boundary {n}+{other}");
        }
        if n.abs() <= i32::MAX as i128 {
            let got = negate_ternary_tq10_10(n as i32).unwrap();
            assert_eq!(got as i128, -n);
        }
    }
    // All-(+1) run IS the maximal balanced remainder — the theorem's edge.
    for m in 1..=8u32 {
        let run = (3i128.pow(m) - 1) / 2;
        let (rounded, r) = ref_round_nearest(run, m);
        assert_eq!(rounded, 0, "max tail must round to zero (nearest)");
        assert_eq!(r, run);
    }
}

// ============================================================================
// Failure modes: loud, never wrapped (contract §3, §5)
// ============================================================================

#[test]
fn overflow_and_domain_failures() {
    assert!(add_ternary_tq10_10(i32::MAX, 1).is_err());
    assert!(subtract_ternary_tq10_10(i32::MIN, 1).is_err());
    assert!(negate_ternary_tq10_10(i32::MIN).is_err());
    assert!(multiply_ternary_tq10_10(i32::MAX, i32::MAX).is_err());
    assert!(divide_ternary_tq10_10(1, 0).is_err());
    // Near-boundary successes stay exact.
    assert_eq!(add_ternary_tq10_10(i32::MAX - 1, 1).unwrap(), i32::MAX);
    assert_eq!(negate_ternary_tq10_10(i32::MIN + 1).unwrap(), i32::MAX);
}

// ============================================================================
// mul3 / div3 shift semantics (contract §3, truncation pin)
// ============================================================================

#[test]
fn mul3_div3_shift_semantics() {
    use g_math::fixed_point::domains::balanced_ternary::TernaryTier1;
    // mul3 ∘ div3 is identity on multiples of 3; div3 truncates toward zero
    // otherwise — pinned as documented current behavior (contract §3):
    // a true ternary shift would round 2 → 1 (nearest), not 2 → 0.
    let t = TernaryTier1::one(); // raw = 3^8
    let up = t.mul3().unwrap();
    let back = up.div3();
    assert_eq!(back, t);
    let two = TernaryTier1::from_integer(2).unwrap();
    let d = two.div3();
    let redecoded = d.mul3().unwrap();
    // 2·3^8 / 3 = 4374 exactly (2·3^8 ≡ 0 mod 3): still exact here.
    assert_eq!(redecoded, two);
}

// ============================================================================
// Wide-tier coverage (gap-closing, post-0.4.33): Tiers 2–6 arithmetic against
// exact integer models, adversarial NEGATIVE operands prioritized — the
// historical wide-integer defect class is unsigned word arithmetic that
// corrupts signed products (mul_to_i512/i1024/i2048 family).
// ============================================================================

use g_math::fixed_point::domains::balanced_ternary::{
    add_ternary_tq20_20, add_ternary_tq40_40,
    subtract_ternary_tq20_20, subtract_ternary_tq40_40,
    multiply_ternary_tq20_20, multiply_ternary_tq40_40,
    divide_ternary_tq20_20, divide_ternary_tq40_40,
    negate_ternary_tq20_20, negate_ternary_tq40_40,
    multiply_ternary_tq80_80, multiply_ternary_tq80_80_checked,
    multiply_ternary_tq160_160, multiply_ternary_tq320_320,
    divide_ternary_tq80_80, divide_ternary_tq160_160, divide_ternary_tq320_320,
    negate_ternary_tq80_80, negate_ternary_tq160_160, negate_ternary_tq320_320,
    TernaryTier4, TernaryTier5, TernaryTier6,
    SCALE_TQ20_20, SCALE_TQ40_40,
};

/// Signed integer pairs with every sign combination — the adversarial axis.
const INT_PAIRS: &[(i64, i64)] = &[
    (7, 5), (-7, 5), (7, -5), (-7, -5),
    (1, 1), (-1, 1), (-1, -1),
    (3280, 3), (-3280, 3), (3280, -3), (-3280, -3),
    (43_046_721, 2), (-43_046_721, 2),          // 3^16 boundary
    (1_853_020, -981), (-1_853_020, -981),
    (123_456_789, -987), (-123_456_789, 987),
];

#[test]
fn tier2_ops_match_exact_i128_model() {
    let scale = SCALE_TQ20_20 as i128;
    for &(a, b) in INT_PAIRS {
        let (ra, rb) = (a * SCALE_TQ20_20 / 1, b * SCALE_TQ20_20 / 1);
        assert_eq!(add_ternary_tq20_20(ra, rb).unwrap() as i128, ra as i128 + rb as i128);
        assert_eq!(subtract_ternary_tq20_20(ra, rb).unwrap() as i128, ra as i128 - rb as i128);
        assert_eq!(negate_ternary_tq20_20(ra).unwrap(), -ra);
        let model = nearest_symmetric(ra as i128 * rb as i128, scale);
        if model >= i64::MIN as i128 && model <= i64::MAX as i128 {
            assert_eq!(
                multiply_ternary_tq20_20(ra, rb).unwrap() as i128,
                model,
                "tier2 mul at ({a},{b})"
            );
        } else {
            // TQ20.20 range: products beyond i64 must fail loud, never wrap.
            assert!(
                multiply_ternary_tq20_20(ra, rb).is_err(),
                "tier2 mul at ({a},{b}) must overflow loud"
            );
        }
        if rb != 0 {
            assert_eq!(
                divide_ternary_tq20_20(ra, rb).unwrap() as i128,
                nearest_ties_up(ra as i128 * scale, rb as i128),
                "tier2 div at ({a},{b})"
            );
        }
    }
}

#[test]
fn tier3_ops_match_exact_i128_model() {
    let scale = SCALE_TQ40_40; // i128
    for &(a, b) in INT_PAIRS {
        let ra = a as i128 * scale;
        let rb = b as i128 * scale;
        assert_eq!(add_ternary_tq40_40(ra, rb).unwrap(), ra + rb);
        assert_eq!(subtract_ternary_tq40_40(ra, rb).unwrap(), ra - rb);
        assert_eq!(negate_ternary_tq40_40(ra).unwrap(), -ra);
        // Integer-lattice product: (a·S)(b·S)/S = a·b·S exactly.
        assert_eq!(
            multiply_ternary_tq40_40(ra, rb).unwrap(),
            a as i128 * b as i128 * scale,
            "tier3 mul at ({a},{b})"
        );
        if rb != 0 && a % b == 0 {
            assert_eq!(
                divide_ternary_tq40_40(ra, rb).unwrap(),
                (a / b) as i128 * scale,
                "tier3 exact div at ({a},{b})"
            );
        }
    }
    // Nearest ties-+∞ semantics at tier 3 (0.5.0): 3.5 raw units is a
    // genuine tie — +3.5 → 4 (up), −3.5 → −3 (toward +∞).
    let seven = 7i128 * scale;
    let two = 2i128 * scale;
    let base = 7i128 * scale / 2; // floor(3.5·S) — S odd, so 7S/2 truncates
    assert_eq!(divide_ternary_tq40_40(seven, two).unwrap(), base + 1);
    assert_eq!(divide_ternary_tq40_40(-seven, two).unwrap(), -base);
}

#[test]
fn tier4_integer_lattice_all_sign_combinations() {
    for &(a, b) in INT_PAIRS {
        let ta = TernaryTier4::from_integer(a as i128);
        let tb = TernaryTier4::from_integer(b as i128);
        let expected = TernaryTier4::from_integer(a as i128 * b as i128);
        let got = multiply_ternary_tq80_80(*ta.raw(), *tb.raw());
        assert_eq!(got, *expected.raw(), "tier4 mul at ({a},{b})");
        let got_checked = multiply_ternary_tq80_80_checked(*ta.raw(), *tb.raw()).unwrap();
        assert_eq!(got_checked, *expected.raw(), "tier4 checked mul at ({a},{b})");
        if b != 0 && a % b == 0 {
            // (a·b)/b == a and (a)/b == a/b, exactly on the integer lattice.
            assert_eq!(
                divide_ternary_tq80_80(*expected.raw(), *tb.raw()),
                *ta.raw(),
                "tier4 div (a*b)/b at ({a},{b})"
            );
            let q = TernaryTier4::from_integer((a / b) as i128);
            assert_eq!(
                divide_ternary_tq80_80(*ta.raw(), *tb.raw()),
                *q.raw(),
                "tier4 div a/b at ({a},{b})"
            );
        }
        // Negation total on the lattice, exact.
        assert_eq!(
            negate_ternary_tq80_80(*ta.raw()),
            *TernaryTier4::from_integer(-(a as i128)).raw(),
            "tier4 neg at {a}"
        );
    }
}

#[test]
fn tier5_integer_lattice_all_sign_combinations() {
    for &(a, b) in INT_PAIRS {
        let ta = TernaryTier5::from_integer(a as i128);
        let tb = TernaryTier5::from_integer(b as i128);
        let expected = TernaryTier5::from_integer(a as i128 * b as i128);
        let got = multiply_ternary_tq160_160(ta.raw().clone(), tb.raw().clone())
            .expect("tier5 lattice mul must not overflow");
        assert_eq!(got, *expected.raw(), "tier5 mul at ({a},{b})");
        if b != 0 && a % b == 0 {
            let q = TernaryTier5::from_integer((a / b) as i128);
            assert_eq!(
                divide_ternary_tq160_160(ta.raw().clone(), tb.raw().clone())
                    .expect("tier5 lattice div must not overflow"),
                *q.raw(),
                "tier5 div at ({a},{b})"
            );
        }
        assert_eq!(
            negate_ternary_tq160_160(ta.raw().clone()).expect("tier5 neg"),
            *TernaryTier5::from_integer(-(a as i128)).raw(),
            "tier5 neg at {a}"
        );
    }
}

#[test]
fn tier6_integer_lattice_all_sign_combinations() {
    // Tier 6 multiply routes through mul_to_i2048 — the widening multiply
    // family with the historical unsigned-word negative-input defect. Every
    // sign combination here is the adversarial case.
    for &(a, b) in INT_PAIRS {
        let ta = TernaryTier6::from_integer(a as i128);
        let tb = TernaryTier6::from_integer(b as i128);
        let expected = TernaryTier6::from_integer(a as i128 * b as i128);
        let got = multiply_ternary_tq320_320(ta.raw().clone(), tb.raw().clone());
        assert_eq!(got, *expected.raw(), "tier6 mul at ({a},{b})");
        if b != 0 && a % b == 0 {
            let q = TernaryTier6::from_integer((a / b) as i128);
            assert_eq!(
                divide_ternary_tq320_320(ta.raw().clone(), tb.raw().clone()),
                *q.raw(),
                "tier6 div at ({a},{b})"
            );
        }
        assert_eq!(
            negate_ternary_tq320_320(ta.raw().clone()),
            *TernaryTier6::from_integer(-(a as i128)).raw(),
            "tier6 neg at {a}"
        );
    }
}
