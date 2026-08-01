# Balanced Ternary Contract

Formal specification of the balanced-ternary domain
(`src/fixed_point/domains/balanced_ternary/`), written against the code as it
ships. Every claim here is either (a) verified by
`tests/ternary_domain_validation.rs`, (b) marked **[current behavior]** for
implementation choices that are documented rather than idealized, or (c)
marked **[theorem]** for mathematical facts the tests prove independently of
the implementation. Nothing in this document describes behavior the suite
does not check.

## 1. Representation law

The domain stores **scaled integers**, not trit vectors:

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

The trit view is a *derived* representation: any raw integer has a unique
canonical balanced-ternary digit expansion (digits in {−1, 0, +1}), and the
oracle in the validation suite converts between the two forms exhaustively.
Packed trit encodings exist only in `trit_q1_9.rs` / `trit_packing.rs`
(the TQ1.9 inference formats, covered by their own suite).

**Range honesty.** Doc comments describe tier ranges by nominal trit window
(e.g. Tier 1 "±(3^8−1)/2"), but the implementation bounds values by *binary
storage*, not by trit count: `TernaryTier1::from_integer` accepts any i16
whose scaled raw fits i32, which exceeds the 8-integer-trit window by ~100×.
The representable set of a tier is exactly

```
{ n / 3^F : n ∈ [STORAGE_MIN, STORAGE_MAX] }
```

The nominal trit window describes precision structure, not an enforced
invariant. **[current behavior]**

## 2. Exactness class

A rational p/q (lowest terms) is exactly representable at scale 3^F iff
`q | 3^F`, i.e. iff q's prime set ⊆ {3} — the ternary node of the router's
exactness lattice (binary {2}, decimal {2,5}, ternary {3}, symbolic ⊤).

Consequence at the domain boundary: **binary-exact values are generally
ternary-inexact**, and the flagship case is ½. Since every scale 3^F is odd,
`½ · 3^F` is never an integer and sits *exactly halfway* between the two
neighboring grid points — conversion into ternary is where genuine rounding
ties live (§5). Inside the domain no such tie can occur (§4).

## 3. Operation semantics (verified against the shipping code)

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
  (e.g. ½) requires a tie rule; the FASC cross-domain path handles such
  values by classification (ternary-exact values only — currently routed to
  Binary; see ROADMAP 0.4.34), so no silent tie-breaking ships today.
  Any future direct converter MUST document its tie rule; this contract
  reserves **ties toward +∞** as the default (matching the binary
  round-bit convention used in `downscale_to_storage`).
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

Suite: `tests/ternary_domain_validation.rs` (profile-independent — Tier 1/2
functions are plain integer ops, no `table_format` gating). UGOD tier
promotion, canonical↔imperative path equivalence, and cross-domain coercion
are items 5–6 of the 0.4.33 plan, tracked in ROADMAP.

## Disclaimer

This software is provided "as is", without warranty of any kind. See the
repository LICENSE and the README disclaimer.
