# Decimal contract-validation harness — design

Status: **Phase 1 (proof of concept)**. Branch `validation/decimal-scaled`.

## Why this exists

GitHub issue feedback (mootable/decimal-scaled, "jackmoxley"): a transcendental
in a fixed-size format cannot be "0 ULP" against the *true* value — the best
achievable is the correctly-rounded nearest representable value (≤ 0.5 ULP).
gMath agrees. This harness proves, per push, exactly how close gMath lands —
measured against a high-precision mpmath oracle, graded under **gMath's own
declared rounding rule**, with honest reporting of where it misses.

It is deliberately *not* a unit test. It is a contract gate with a report.

## The oracle format (adopted from decimal-scaled, independently generated)

Each golden line is mode-agnostic:

    <input_raw>\t<floor_raw>\t<cls>          # unary
    <a_raw>\t<b_raw>\t<floor_raw>\t<cls>     # binary (phase 2)

- `floor_raw` = floor(f(x)·10^scale) toward −∞ (sign lives here).
- `cls` ∈ {Z=exact, L=below-half, E=exact-tie, G=above-half}, frac in [0,1).

From `(floor, cls, sign, RoundingMode)` the correctly-rounded result is
`floor` or `floor+1` with **zero tolerance**. One table grades every rounding
mode. We use mpmath ≥ `2·scale+80` digits of guard precision to classify.

We generate our **own** fixtures (oracle #1, committed, small) and can grade
against decimal-scaled's committed corpus (oracle #2, env-gated, not vendored —
`GMATH_DECIMAL_SCALED_GOLDEN` points at a local clone pinned to e6c7497).

## The grader (ported to agree with decimal-scaled bit-for-bit)

`oracle_correctly_rounded(floor, cls, mode)` mirrors decimal-scaled's
`bump_to_ceil` + half-even parity rule exactly, so our pass/fail agrees with
Jack's. A dedicated unit test pins the full (cls × mode × sign) truth table.

## The rounding finding (load-bearing)

decimal-scaled is a **finance** library; its default rounding is **HalfToEven**
(banker's, unbiased — sums don't drift). gMath's decimal domain rounds
**HalfAwayFromZero** (commercial; biased upward on ties).

Consequences:
1. **Fairness:** gMath decimal output is graded under HalfAwayFromZero — its
   *actual* rule — not Jack's half-even default. Otherwise every exact tie
   fails spuriously.
2. **Gap metric:** the harness counts, per function, how often half-away and
   half-even disagree (the tie-divergence count). This quantifies the
   financial-rounding gap and motivates a future `_with(RoundingMode)` decimal
   path (HalfToEven) — the highest-value next-release item for finance use.

## The decimal domain has TWO public surfaces (verified in code)

The transcendental engine (`decimal_exp/ln/sqrt/sin` in `transcendental/*.rs`,
operating on `ComputeStorage`) is a **standalone shared core**. Two public
wrappers call it, and they materialize at **different scales**:

| Surface (arm) | Entry | Final scale (balanced) | Extraction |
|---------------|-------|------------------------|-----------|
| A imperative  | `DecimalFixed::<N>::exp()` | exactly `N` | `raw_value()` |
| B canonical   | `gmath("x").exp()` + `evaluate()` | `DecimalCompute` at compute dp (77); materializes to `DECIMAL_STORAGE_MAX_DP-2` (=36) on Display/`to_rational` | downscale the `DecimalCompute` |

They SHARE the engine and the round-half-away-from-zero rule but DIVERGE on
final materialization scale, so they are not bit-identical. `DecimalTier1..6`
are UGOD overflow/arithmetic tiers (no transcendentals) — not a third engine.

Phase 1 proves Arm A (deterministic scale, clean `raw_value()` extraction).
Arm B (the canonical surface most users hit, scale 36) shares the engine but
has its own output-rounding scale and gets its own arm next — same generator
at `SCALE=36`, same grader.

## Per-domain plan (his inputs are decimal-scaled integers)

| gMath domain        | input rep error | grading rule        | phase |
|---------------------|-----------------|---------------------|-------|
| Decimal (DecimalFixed<S>) | zero      | half-away-from-zero | **1** |
| Binary (Q-format)   | nonzero         | ties-toward-+∞      | 2     |
| Balanced ternary    | nonzero         | (via binary engine) | 2     |
| Symbolic/rational   | zero (arith)    | exact (Z) expected  | 2     |
| Fractal router      | —               | domain-selection + exactness | 2 |

Hard constraint: gMath `DecimalFixed` is i128-backed (caps ~d38). gMath's
"go wider" answer is the **binary** domain (Q64.64–Q256.256); decimal-scaled's
is decimal (to d1232). This is the genuine philosophical fork — the harness
shows it rather than hiding it.

`DecimalFixed<S>` must run on a profile whose decimal compute dp exceeds S
(guard digits). Scale 28 requires balanced (compute dp 77) or scientific;
embedded (compute dp 38) leaves too little guard for wide scales.

## Phase 1 scope (this commit)

- Functions: exp, ln, sqrt, sin (covers large-monotone, domain-restricted,
  algebraic-exact, periodic).
- One domain (decimal), one profile (balanced), scale 28.
- Own mpmath fixtures + grader + report + honest buckets:
  correctly-rounded / tie-class(E) separate / out-of-range skipped /
  contract-violation (the only CI-failing bucket).
- decimal-scaled oracle path wired but env-gated (off unless the var is set).

Out of scope until later phases: other functions, other domains, the router,
multi-profile CI matrix, the live decimal-scaled crate as a peer (awaits Jack's
hook crate).
