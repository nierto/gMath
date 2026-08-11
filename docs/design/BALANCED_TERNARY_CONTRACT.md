# Balanced Ternary Contract

Formal specification of the balanced-ternary domain
(`src/fixed_point/domains/balanced_ternary/`), written against the code as it
ships. Every claim here is either (a) verified by
`tests/ternary_domain_validation.rs`, (b) marked **[current behavior]** for
implementation choices that are documented rather than idealized, or (c)
marked **[theorem]** for mathematical facts the tests prove independently of
the implementation. Nothing in this document describes behavior the suite
does not check.

## 1. Representation law — two coexisting forms

The domain carries balanced ternary in **two distinct representations**, and
which invariants apply depends on which one you are holding.

### 1a. Native trits (storage + inference path)

Genuinely native balanced ternary: the digits *are* the representation and
the arithmetic exploits {−1, 0, +1} directly.

- `Trit` — `#[repr(i8)] enum { Neg = -1, Zero = 0, Pos = 1 }`
  (`trit_packing.rs`).
- `pack_trits` / `unpack_trits` — 5 trits per byte, base-3
  (`d₀·81 + d₁·27 + d₂·9 + d₃·3 + d₄`, digits offset +1 for unsigned
  storage), 1.6 bits/trit against the 1.585 information-theoretic minimum.
- `trit_dot`, `packed_trit_dot`, `packed_trit_matvec` — **zero-multiply**
  arithmetic: the inner loop branches on the trit and accumulates
  `acc ± activation`, never a multiply. `TRIT_DECODE_TABLE` unpacks a
  byte into 5 trits by table lookup; an AVX2 path exists for the realtime
  profile. This is the Setun property doing real work — a ternary digit
  needs no multiplier.

### 1b. Scaled integers (UGOD tier arithmetic path)

The six UGOD tiers store a radix-3 **scaled integer** in binary storage:

```
value = raw / 3^F        raw ∈ native signed integer storage
```

| Tier | Format   | F (frac trits) | Scale  | Storage |
|------|----------|----------------|--------|---------|
| 1    | TQ8.8    | 8              | 3^8    | i32     |
| 2    | TQ16.16  | 16             | 3^16   | i64     |
| 3    | TQ32.32  | 32             | 3^32   | i128    |
| 4    | TQ64.64  | 64             | 3^64   | I256    |
| 5    | TQ128.128| 128            | 3^128  | I512    |
| 6    | TQ256.256| 256            | 3^256  | I1024   |

Here the arithmetic is binary integer arithmetic on `raw` (add is
`checked_add`; multiply is a double-width product rescaled by 3^F). The
trit expansion still exists and is unique — every integer has exactly one
canonical balanced-ternary form — but it is *derived*, not stored, and no
tier operation materializes it. §4's theorem is what guarantees the two
views agree; `theorem_trit_truncation_is_round_nearest` proves that
agreement exhaustively (the suite's oracle is itself a native trit
implementation, so the test bridges 1a and 1b directly).

### 1c. TQ1.9 — scaled, but trit-window enforced

`TritQ1_9 { raw: i16 }` stores `value × 3^9` (scaled, like 1b) but —
unlike the UGOD tiers — **enforces the 10-trit window**:
`MAX_RAW = (3^10 − 1)/2 = 29524` is range-checked in every constructor,
conversion, and arithmetic method. Its weights are repacked into the
native-trit form of 1a for the zero-multiply matvec paths.

**Range honesty (1b only).** Tier doc comments describe ranges by nominal
trit window (Tier 1: "±(3^8−1)/2 ≈ ±3280"), but the UGOD tiers bound values
by *binary storage*, not trit count. The representable set of a tier is
exactly

```
{ n / 3^F : n ∈ [STORAGE_MIN, STORAGE_MAX] }
```

For Tier 1 that admits values up to `(2^31−1)/3^8 ≈ 327,310` — about **100×**
the nominal ±3280 window (and `from_integer`, capped by its `i16` parameter,
still reaches ±32,767, about **10×** the window). For the UGOD tiers the
nominal trit window describes precision structure, not an enforced
invariant. TQ1.9 (1c) is the exception that does enforce it.
**[current behavior]**

## 2. Exactness class

A rational p/q (lowest terms) is exactly representable at scale 3^F iff
`q | 3^F`, i.e. iff q's prime set ⊆ {3} — the ternary node of the router's
exactness lattice (binary {2}, decimal {2,5}, ternary {3}, symbolic ⊤).

Consequence at the domain boundary: **binary-exact values are generally
ternary-inexact**, and the flagship case is ½. Since every scale 3^F is odd,
`½ · 3^F` is never an integer and sits *exactly halfway* between the two
neighboring grid points — conversion into ternary is where genuine rounding
ties live (§5). Inside the domain no such tie can occur (§4).

## 3. Operation semantics — UGOD tiers (§1b)

The table below covers the **scaled-integer tier arithmetic**. The
native-trit operations of §1a have their own contract: they are exact
integer accumulations at compute tier with a single narrowing (validated
by `tests/tq19_validation.rs` and the `matvec_q2f` wide-output property
tests), and being multiply-free they introduce no rescaling error at all.

| Op | Semantics | Error | Failure mode |
|----|-----------|-------|--------------|
| add / sub | exact | 0 | `checked_add/sub` → `TierOverflow` at storage bound |
| neg | exact | 0 | `checked_neg` → `TierOverflow` (binary MIN only; unreachable from any negatable value) |
| mul | `(a·b) / 3^F` in a double-width intermediate, **truncated toward zero** | < 1 ulp, toward zero | range check → `TierOverflow` |
| div | `(a·3^F) / b` in a double-width intermediate, **truncated toward zero** | < 1 ulp, toward zero | `DivisionByZero`; range check → `TierOverflow` |
| mul3 | exact ternary up-shift | 0 | `checked_mul` → overflow error |
| div3 | `raw / 3`, truncated toward zero | < 1 ulp | infallible |

Notes, all **[current behavior]**:

- Toward-zero truncation is **odd-symmetric**: `mul(−a, b) = −mul(a, b)` and
  `div(−a, b) = −div(a, b)` exactly, because Rust integer division truncates
  toward zero. The suite pins this exhaustively.
- Toward-zero is *not* the balanced-ternary natural rounding (§4). It is,
  however, deterministic, symmetric, and strictly sub-ulp.
- `div3`'s doc comment says "exact division by 3"; it is exact only when
  `raw ≡ 0 (mod 3)` and otherwise truncates toward zero (`raw = 2 → 0`).
  A true ternary shift (drop lowest trit) would round to *nearest*
  (`raw = 2 → 1`, since 2/3 is nearer 1). Flagged; behavior unchanged.
- Tier 4 negation uses `saturating_neg` where every other tier uses
  `checked_neg` — inconsistent failure mode at the (unreachable-in-practice)
  storage MIN. Flagged; behavior unchanged.

## 4. The tie-free rounding theorem **[theorem]**

> **Theorem.** For any integers n and m ≥ 1, the halfway point between
> consecutive multiples of 3^m is never an integer, because 3^m is odd.
> Therefore round-to-nearest of n onto the grid 3^m·Z is **total without any
> tie-breaking rule**: the nearest multiple is unique, and the balanced
> remainder r = n − 3^m·q, normalized into [−(3^m−1)/2, +(3^m−1)/2],
> satisfies |r| ≤ (3^m−1)/2 < 3^m/2 **strictly**.

Equivalently in digit form: truncating (dropping) the lowest m balanced
trits of n's canonical expansion yields exactly this nearest multiple — the
discarded tail is bounded by Σ|dᵢ|·3^i ≤ (3^m−1)/2. **Trit truncation IS
round-to-nearest**, and no tie case exists to break.

Corollaries, each pinned by a dedicated test:

1. `round_nearest(−n) = −round_nearest(n)` — symmetry is free, no
   ties-to-even / ties-away distinction can arise (the modes coincide
   vacuously).
2. Balanced-ternary rounding is unbiased with no tie-handling hardware or
   logic — the property that made truncation safe on Setun.
3. The two candidate "ties to even" definitions sometimes proposed (last
   retained trit = 0 vs. retained coefficient divisible by 3) are the same
   condition and both moot: Σdᵢ3^i ≡ d₀ (mod 3).

**Scope.** The theorem covers rounding *within* the ternary grid family
(3-adic scale changes: mul/div rescaling, tier demotion, trit truncation).
It does **not** cover:

- **Conversion in** from another domain: binary ½ = 0.111…₃ (repeating) is
  exactly equidistant from its two ternary neighbors at every scale — an
  exact tie requiring an explicit rule at the boundary (§5).
- **Division**, whose exact quotients are generally not on any 3-adic grid.

## 5. Boundary rules

- Conversion into ternary from values with denominators outside {3}
  (e.g. ½) requires a tie rule. The shipping direct converter
  (`UniversalTernaryFixed::from_str`, used by `0t` literals) **truncates
  toward zero** — measured and pinned: `0.5 → 3280/3^8` (the tie resolves
  low), `1.5 → 9841/3^8` — and is sign-symmetric
  (`parse(-s) = -parse(s)`; the `-0.x` sign-loss defect was fixed in
  0.4.33). Any future converter with different rounding MUST document its
  tie rule.
- Tier raws must fit the profile's FASC storage: `ternary_to_storage` is
  checked (0.4.33) and returns `TierOverflow` for a Tier-2+ raw on a
  profile whose `BinaryStorage` cannot hold it, where it previously
  wrapped silently. Values *reached by arithmetic* stay at their operand
  tier, so e.g. `0t3280 + 0t1` is representable everywhere while the
  literal `0t3281` window-gates to Tier 2 and errors loudly on realtime —
  an asymmetry pinned by test.
- Storage overflow is always an error (`TierOverflow`), never a wrap —
  the wrap-defect rule applies to this domain as everywhere else.

## 6. Current behavior vs. theorem — the deliberate gap

mul/div/div3 truncate toward zero (< 1 ulp) instead of the tie-free
balanced-nearest (< ½ ulp) that §4 makes available at one extra comparison
per operation. Switching would change results (a breaking precision change)
and is therefore an **owner decision, out of scope for 0.4.33**. The suite
tests what ships; the theorem tests prove the upgrade path exists and needs
no tie logic. This gap is the concrete candidate for 0.4.34+ alongside the
routing column.

## 7. Invariant checklist (test map)

| Invariant | Test |
|---|---|
| encode/decode roundtrip, canonical digits ∈ {−1,0,+1} | `oracle_roundtrip_exhaustive`, `oracle_digits_balanced` |
| trit-wise add with carry ≡ raw add | `oracle_add_matches_raw_add` |
| trit flip ≡ negation; `neg(neg(x)) = x`; `x + (−x) = 0` | `oracle_neg_matches_raw_neg`, `negation_involution_and_inverse` |
| trit shift-and-add mul ≡ exact product | `oracle_mul_matches_exact_product` |
| tq8_8 mul/div = toward-zero, odd-symmetric | `mul_div_toward_zero_and_symmetric` |
| balanced remainder bound, strict half-ulp, no tie | `theorem_no_ties_balanced_remainder` |
| trit-drop = unique nearest multiple | `theorem_trit_truncation_is_round_nearest` |
| `round(−n) = −round(n)` | `theorem_rounding_symmetry` |
| ½ conversion tie (odd scale equidistance) | `boundary_half_is_exact_tie` |
| boundary families ±3^k, ±(3^k±1), trit runs, alternating | `boundary_families` |
| overflow/domain failures loud, never wrapped | `overflow_and_domain_failures` |
| mul3/div3 shift semantics incl. truncation pin | `mul3_div3_shift_semantics` |
| UGOD promotion on raw overflow, value-exact | `ugod_promotion_on_multiply_overflow`, `ugod_promotion_preserves_value_exactly` |
| mixed-tier alignment; from_str window gate | `ugod_mixed_tier_alignment`, `ugod_window_boundaries_exact` |
| FASC `0t` arithmetic ≡ imperative UGOD | `fasc_ternary_arithmetic_stays_ternary_and_exact`, `fasc_ternary_negation_symmetry` |
| cross-domain coercion is value-neutral | `cross_domain_coercion_matches_plain_expressions` |
| conversion truncation + sign symmetry pins | `fractional_literal_conversion_boundary`, `negative_fractional_literal_sign_regression` |
| storage narrowing loud on narrow profiles | `ternary_literal_tier2_storage_limit_is_loud` |

Suites: `tests/ternary_domain_validation.rs` (oracle + theorem tests,
profile-independent) and `tests/ternary_path_equivalence.rs` (UGOD
promotion, FASC↔imperative equivalence, coercion, conversion pins) — both
run per-push by the `ternary-domain` CI workflow on realtime + compact.
The oracle being a native trit implementation means the suite validates
the §1b scaled-integer path *and* its agreement with the §1a trit view.
The §1a operations themselves (packing, zero-multiply dots, matvec) are
covered by `tests/tq19_validation.rs` and the `fused-tq19-precision`
workflow.

## Disclaimer

This software is provided "as is", without warranty of any kind. See the
repository LICENSE and the README disclaimer.
