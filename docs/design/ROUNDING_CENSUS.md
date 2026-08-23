# Rounding-Site Census (0.5.0 item 0c)

**STATUS: IMPLEMENTED 2026-08-23 (stage b).** Every "Change for 0c" row
below has landed; `tests/rounding_unification.rs` gates cross-path
bit-equality (sweeps + constructed exact ties) per profile, and all five
profiles pass their full suites. Corrections discovered during
implementation:

- Canonical decimal divide tiers 1–5 were NOT truncating — they are
  **exact-or-rational-fallback** (`PrecisionLoss` → symbolic), which is
  better than any rounding rule and was kept; only the tier-6 best-effort
  arm truncated, now banker's.
- `decimal_to_binary_storage` truncated on four arms and add-half-rounded
  on q16_16 — unified to nearest ties-+∞ (result domain rule).
- The UGOD binary divide's old half-away adjust also mis-signed its bump
  for exact quotients in (−1, 0) raw units (branched on `quotient < 0`,
  which is 0 there) — fixed by deriving the sign from the operands.
- Ternary Tier-4 `saturating_neg` → fail-loud MIN assert (0.4.33 flag).
- LU singularity nuance surfaced: exact-zero pivots for matrices with
  storage-inexact multipliers (e.g. pivoting on 7 → 1/7) were a
  truncation-cancellation coincidence; nearest leaves honest ulp noise.
  Tests updated: dyadic-multiplier matrices must still fail loud, the
  non-dyadic classic asserts an ulp-noise determinant bound.

Original census (stage a) follows.

Every rounding site that determines result bits, catalogued from source
2026-08-14. This is the evidence base for the uniform rounding policy
(ROADMAP 0.5.0 item 0c): **exact when representable, round once otherwise,
one tie rule per domain across every path** — binary nearest-ties-up,
decimal banker's, ternary nearest (tie-free by theorem).

Method: exhaustive grep for `round_bit` / `banker` / bare-shift /
truncating-division patterns, then source reads of every hit. Sites marked
**[verified]** were read line-by-line; a handful are **[catalogued]** —
located but their rule pinned down during implementation.

## A. Storage-result rounding sites (the policy surface)

### Binary domain — target: round-to-nearest, ties toward +∞

| Site | Current rule | Change for 0c |
|---|---|---|
| `fixed_multiply` (imperative `FixedPoint::Mul`) q16_16/q32_32 arms | **floor** (bare arithmetic `>>`) [verified] | → nearest ties-up (up to 1 ulp on ~all inexact products — behavioral improvement, breaking) |
| `fixed_multiply` q64_64 arm (`multiply_binary_i128` + AVX2 twin) | **banker's** (both twins consistent) [verified] | → ties-up (exact ties only) |
| `fixed_multiply` q128_128/q256_256 arms | **truncate toward zero** (sign-magnitude shift) [verified] | → nearest ties-up (up to 1 ulp, breaking) |
| `fixed_divide` (imperative `FixedPoint::Div`) — ALL profiles | **truncate toward zero** (bare `/`) [verified] | → nearest ties-up (up to 1 ulp on ~all inexact quotients, breaking) |
| UGOD `BinaryTier1..6::checked_mul`/`mul` | nearest **ties-up** (round-bit add, all 6 tiers) [verified] | none — already the target |
| UGOD `BinaryTier*::checked_div` | nearest **half-away** (`abs_2rem >= abs_div`) [verified] | → ties-up (exact ties only) |
| `downscale_to_storage` / `round_to_storage` / `downscale_q64_to_q32` (every compound path's single rounding) | nearest **ties-up**, checked/saturating [verified] | none — this is the anchor the policy aligns TO |
| `compute_tier::to_fixed` (0.4.32 public) | ties-up via `downscale_to_storage` [verified] | none |

**Headline finding: the imperative binary multiply uses THREE different
rules by profile** (floor / banker's / toward-zero), and the imperative
divide truncates everywhere — CONTRACT.md §3 was still wrong after the
2026-08-14 correction, which had only checked the embedded kernel. §3 is
re-corrected alongside this census.

**Measured consequence** (compact, 44,044 sampled raw pairs): imperative
vs canonical storage-tier multiply differ on **48.7% of products**, max
1 ulp — floor vs nearest diverges whenever the discarded bits are >= half
an ulp, i.e. on ~half of all inexact products, NOT just on exotic ties.
The same argument applies to divide on every profile (truncate vs
half-away). Path independence for direct storage-tier mul/div is
therefore materially broken today on most profiles; compound/tier-N+1
paths are unaffected (single shared downscale). This is the strongest
motivation for 0c: unification does not just simplify the mental model,
it REPAIRS the path-independence contract for plain arithmetic.

### Decimal domain — target: banker's (half-even)

| Site | Current rule | Change for 0c |
|---|---|---|
| `DecimalFixed<D>` mul + div (11 `banker_round_decimal_i128` sites) | **banker's** [verified] | none — already the target |
| Canonical decimal multiply (`try_mul_exact`) | **exact** — decimal places grow, no rounding [verified] | none — exactness-first, keep |
| Canonical decimal divide | **truncate** (no rounding code; tier-6 comment confirms) [verified] | → banker's |
| dp-cap reduction on canonical decimal store/promote | [catalogued] — pin rule during 0c | → banker's |
| `decimal_to_binary_storage` (cross-domain coercion) | [catalogued] — 0.3.90 notes say round-to-nearest; tie rule unpinned | → domain of RESULT (binary ties-up) |
| Decimal compute-tier engine mul (`decimal_compute.rs`) | half-away at DECIMAL_COMPUTE_DP [verified] | engine-internal, absorbed (below storage ulp) — keep, note only |

### Ternary domain — target: nearest (tie-free; no rule can be needed)

| Site | Current rule | Change for 0c |
|---|---|---|
| `multiply_ternary_tq*` / `divide_ternary_tq*` (all 6 tiers) | **truncate toward zero** [verified, contract-pinned] | → nearest (< ½ ulp, one comparison; resolves 0.4.33-flagged decision) |
| `div3` | truncate toward zero [verified] | → nearest = true trit shift |
| `convert_to_ternary` / `0t` fractional-literal conversion | truncate toward zero [verified, pinned] | → nearest (tie-free for the 3-adic routed class; conversion-in of non-3-adic values documents nearest with the boundary tie noted in the contract) |

### Contracted exceptions (documented truncation — keep as-is)

| Site | Rule | Why it stays |
|---|---|---|
| tq19 `matvec_q2f`/`wide_output` narrowing | truncation toward zero | published 0.4.31 contract: `q2f / (1<<F)` reproduces the narrow matvec bit-for-bit — nested truncation is the property consumers pinned |

## B. Compute-tier internal roundings (absorbed by tier N+1, out of policy scope)

These round below the storage ulp and are erased by the single downscale;
listed for completeness, no change:

- `compute_divide` — truncating (all profile arms)
- `mul_to_*` wide-product downscales in i512.rs/i1024.rs — ties-up round-bit
- Transcendental engine internals (exp/ln table+Taylor steps, decimal exp
  4-stage, `sincos` reductions) — various, validated end-to-end at 0 ULP
- `compute.rs` 18 round-bit sites — the ties-up downscale family

## Blast-radius summary for 0c implementation

- **Exact-ties-only changes** (astronomically rare inputs): embedded
  imperative mul (banker's→ties-up), UGOD div (half-away→ties-up).
- **Up-to-1-ulp-on-common-inputs changes** (real churn, real improvement):
  imperative mul on realtime/compact (floor→nearest) and
  balanced/scientific (trunc→nearest); imperative div everywhere
  (trunc→nearest); canonical decimal div (trunc→banker's); ternary mul/div
  (trunc→nearest, <½ ulp).
- Consumers freezing hashed/persisted outputs (GeoH-class) must pin
  versions across the 0.5.0 boundary; the imperative-div change is the
  widest-reaching single item and deserves its own line in the notice.
- Every change lands with constructed-tie/cross-path equivalence tests;
  CONTRACT.md §3 collapses to one row per domain when done.
