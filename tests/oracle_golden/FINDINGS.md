# Decimal validation — Phase 1 & 2 findings

## RESOLUTION (imperative composed transcendentals)

The 9 composed imperative `DecimalFixed` methods (tan, asin, acos, sinh, cosh,
tanh, asinh, acosh, atanh) were rewritten to compose at the compute tier (single
downscale), mirroring the binary `FixedPoint` pattern, with the domain/boundary
guards FASC has but the imperative path lacked. `acos` shares an `asin_compute`
core so it never re-rounds at storage tier. `tanh` saturates to ±1 on exp
overflow.

Result across the same 21,884 adversarial inputs: **all 15 native unary
transcendentals now 0 LSB, 0 panics** (was: 2,592 panics, max error 3.5e27,
asinh ~47% wrong). The sweep gate (`gate_policy`) now requires 0 LSB for all 15,
so this is locked against regression.

Root cause confirmed during the fix: `*self * *self` at storage tier overflowed
i128 (e.g. `1.466e19² = 2.15e38 > i128::MAX`) producing garbage; compute-tier
`decimal_compute_mul` widens to I1024 and cannot overflow.

### The "atanh sign bug" was a parser bug (more serious)

Tracing the FASC atanh sign flip led upstream: `gmath_parse` dropped the sign of
**any negative decimal with a zero integer part** — `-0.5` parsed to `+0.5`,
`-0.999` to `+0.999`. Root cause (parsing.rs:142,157): the fractional part's
sign was keyed off `integer_part < 0`, which is false for `-0`, so the negative
sign was silently lost. Fixed to key off the leading-`-` flag (`is_negative`),
matching the already-correct >38-digit path. atanh only surfaced it because its
probe was the one input with |x|<1 and a negative sign; the imperative sweep
never caught it because it uses `from_raw` (a signed i128), not string parsing.
This affected the primary public entry point (`gmath`/`gmath_parse`) for all
sub-unit negatives — e.g. -$0.50. Regression test: `parse_negative_subunit_sign`.

### Both user-reachable paths swept clean

- **Arm A** (imperative `DecimalFixed`): 21,884 inputs, all 0 LSB.
- **Arm B** (FASC canonical `gmath()`): 20,249 inputs, all 0 LSB — the path most
  users hit, with the fractal router picking the domain per input. Previously
  only spot-checked; now fully swept and gated.

The **binary `FixedPoint`** sweep was deliberately dropped: his inputs are
decimal-scaled, so feeding e.g. `0.1` to binary mixes representation error with
engine error inseparably, and the binary engine is already 0-ULP validated
against mpmath directly (`fasc_ulp_validation`). It would only re-demonstrate a
known representation limit — over-engineering.

### 4-column arm (add/sub/mul/div/atan2) + a third bug

decimal-scaled's two-argument golden. Findings: `add`, `sub`, `atan2` correctly
rounded (atan2 1461/1461 exact — decimal atan2 is solid). gMath's decimal mul/div
use **banker's rounding** (already the financial default), so they're graded
under HalfToEven.

**Bug: the `*` / `/` operators overflow the unscaled intermediate.**
`pure_decimal_multiply_optimized_decimal` multiplies in digit-arrays (safe) but
reassembles the unscaled product back into an **i128** before scaling — which
overflows when `|a·b|` exceeds i128::MAX, even though the final result fits.
Example: `1.3333333333333333333 × 7.0 = 9.333…1` — result fits, but the
intermediate `a·b = 9.33e38 > i128::MAX`, so the operator overflows. `div` has
the analogous issue (`a·10^S` overflows for large `a`). The D256-based
`multiply_exact_decimal` does NOT have this — so the fix is to route the
operators through the wide intermediate (or add overflow-detect-and-promote,
UGOD-style). Financially relevant (large value × rate). Gated: add/sub/atan2 at
0; mul/div report-only until fixed.

### Helper renames (clarity)

The domain-polymorphic FASC helpers were misnamed `binary_*` despite preserving
the operand's domain. Renamed: `to_binary_value`→`to_compute_value`,
`binary_divide`→`divide_at_compute`, `halve_binary`→`halve_value`, each with a
doc comment stating PATH membership (FASC composed-transcendental machinery, not
used by the imperative `FixedPoint`/`DecimalFixed` types). `make_binary_int` /
`to_binary_storage` are genuinely binary and kept.

### Symbolic→binary fallback: cost-benefit (Bayesian)

Question: replace the silent Symbolic→binary fallback in `to_compute_value`
with (1) a minimal guard (route Symbolic→decimal compute) or (2) a proper fix
(stay symbolic through the rational-arithmetic sub-steps, approximate only at
the irrational step)?

Findings:
- **Symbolic arithmetic already stays exact** (`add_via_rational` + fractal
  router coercion). No correctness bug. The fallback only affects composed
  *transcendentals*, whose result is irrational and must approximate regardless.
- **Measured accuracy of the current binary fallback** (balanced, vs mpmath,
  full 38-digit storage): atanh(1/3), atanh(1/10), atanh(1/4), asinh(1/7),
  tanh(2/3), acosh(7/3) — all **0 ULP**. Even decimal-exact-but-binary-inexact
  inputs (1/10, 1/4) are correctly rounded: the compute tier (~77 digits)
  absorbs the binary representation error before the downscale to 38 digits.
- **Base rate** of symbolic-operand→composed-transcendental is very low
  (transcendentals are applied to measured decimals/binaries, not exact
  fractions).

Expected accuracy gain = base_rate × per-call-gain ≈ (rare) × (0 measured) ≈ 0.
- **Proper fix (2): REJECT** — over-engineering. ~0 accuracy gain at storage
  tier, but adds BigInt-rational arithmetic on the (rare) symbolic path
  (slower) and real structural complexity/risk.
- **Minimal guard (1): optional, cleanliness-only** — no measurable accuracy
  benefit; its only merit is removing the silent binary drop for domain
  consistency. Thin-guard profiles (realtime/compact) *might* show ≤1 ULP where
  the minimal guard helps, but symbolic transcendentals there are rarer still.

Decision recorded in `to_compute_value`'s doc comment; guard deferred as a
cleanliness nicety, not a correctness or accuracy fix.

---


Branch `validation/decimal-scaled`. Harness: `tests/decimal_contract_validation.rs`.
Oracle #1: our mpmath fixtures (`tests/oracle_golden/*_s28.txt`).
Oracle #2: mootable/decimal-scaled corpus, pinned `e6c7497` (env-gated, not vendored).

## TL;DR

gMath's decimal domain has **two quality tiers**, and the split is caused by
**how each function is composed, not by the domain**:

- **Dedicated native engines** (`exp, ln, sqrt, sin, cos, atan`) compute at the
  compute tier (tier N+1) and downscale once. They are **correctly rounded** —
  0 LSB across ~9,000 adversarial inputs, every scale, 0 panics.
- **Composed transcendentals** (`tan, asin, acos, sinh, cosh, tanh, asinh,
  acosh, atanh`) on the **imperative `DecimalFixed` surface (Arm A)** compose at
  **storage tier** with raw `i128` arithmetic — **bypassing the compute tier and
  UGOD**. Result: 2,592 panics on valid-domain inputs and gross errors
  (`asinh` ~47% wrong on a benign input).
- The **FASC / router path (Arm B)** composes the *same formulas* at the compute
  tier with chain persistence + UGOD, and computes those same inputs
  **correctly to ~36 digits**. The router path is what the design intended; the
  imperative composed methods are the defect.

The router would indeed never route a real computation into the broken code,
and UGOD exists precisely to absorb the overflows the broken path hits — the
imperative composed methods simply do not use either.

## Phase 1 — machinery (oracle #1)

Built the contract harness: independent mpmath generator emitting
decimal-scaled's mode-agnostic `(input_raw, floor_raw, cls)` format; a grader
ported to agree with decimal-scaled bit-for-bit (pinned by a class×mode×sign
truth-table test); honest reporting buckets.

Proven on balanced at `DecimalFixed<28>` (scale 28): exp/ln/sqrt/sin 64/64
correctly rounded, max_delta 0. This validated wiring + grader — but it was a
single, benign cell (thick guard, 4 well-behaved functions). It hid everything
below.

## Phase 2 — full sweep (oracle #2)

Generalized to all 15 native unary transcendentals × decimal-scaled's d18+d38
tiers × every scale = **21,884 adversarial inputs**. Per-function result:

| function | n | exact (0 LSB) | 1 LSB | 2–8 | >8 | panics | max_delta |
|----------|----|--------------|-------|-----|----|--------|-----------|
| exp, ln, sqrt, sin, cos, atan | ~9,000 | **all** | 0 | 0 | 0 | 0 | **0** |
| sinh | 1450 | 1058 | 392 | 0 | 0 | 0 | 1 |
| cosh | 1522 | 1141 | 381 | 0 | 0 | 0 | 1 |
| tan  | 1460 | 511 | 436 | 128 | 63 | 322 | 40382 |
| asin | 1388 | 607 | 440 | 21 | 6 | 314 | 112 |
| acos | 1302 | 580 | 413 | 29 | 7 | 273 | 113 |
| tanh | 1492 | 530 | 557 | 7 | 0 | 398 | 2 |
| asinh| 1573 | 598 | 163 | 57 | 91 | 664 | 5.5e18 |
| acosh| 1321 | 859 | 130 | 41 | 5 | 286 | 2.7e18 |
| atanh| 1353 | 571 | 267 | 69 | 111 | 335 | 3.5e27 |

The harness is sound by internal consistency: the same parser/oracle/grader
gives 0 LSB on six dedicated engines across ~9,000 inputs — impossible if the
machinery were wrong. Oracle was triple-checked (fresh mpmath == decimal-scaled
`floor_raw` exactly on every probed case).

`ties(E)=0` and `fin_gap=0` everywhere: transcendentals are irrational, never
land on an exact half-tie, so half-away-vs-banker's rounding is **moot for
transcendentals** — it is an *arithmetic*-domain concern (money sums), not this.

## Root cause (source: `decimal_fixed.rs:762–867`)

Dedicated engine — correct pattern:

```rust
pub fn sin(&self) -> Self {
    let compute = self.to_decimal_compute();        // -> compute tier (I512 on balanced)
    let result = decimal_sin(compute).expect(...);  // native engine at tier N+1
    Self::from_decimal_compute(result)              // single downscale
}
```

Composed method — defective pattern:

```rust
pub fn asinh(&self) -> Self {
    let x2 = *self * *self;                              // storage-tier i128 mul; x^2 OVERFLOWS i128 near range
    let inner = Self { value: x2.value + Self::SCALE };  // raw i128 add (no checked, no promote)
    let root = inner.sqrt();                             // re-rounds to storage
    let arg = Self { value: self.value + root.value };
    arg.ln()                                             // inner engine .expect() panics on corrupted input
}
pub fn tan(&self)  { Self { value: (s.value * Self::SCALE) / c.value } }  // div-by-0 at poles
pub fn asin(&self) { ... (self.value * Self::SCALE) / denom.value ... }   // div-by-0 at |x|=1
pub fn tanh(&self) { let e2x = two_x.exp(); ... }                        // exp(2x) overflows for large x (should saturate)
```

The composed methods:
1. operate on already-downscaled **storage** values (double rounding / precision
   loss before the formula even starts);
2. do raw `i128` arithmetic (`self.value * Self::SCALE`, `x2.value + SCALE`) that
   **overflows i128** in regions where the I512 compute tier would not — with
   **no overflow detection, no checked ops, no UGOD tier promotion**;
3. divide by near-zero at domain boundaries (`tan` poles, `asin`/`acos` at
   |x|=1, `atanh` at |x|→1) with no guard → panic or blow-up;
4. never call the compute-tier composition the dedicated engines and the FASC
   path use.

## Arm A vs Arm B — the decisive evidence

Same inputs, two surfaces (balanced profile):

| case | Arm A imperative | Arm B FASC/router | truth |
|------|------------------|-------------------|-------|
| asinh(−1.466) | −1.729 (47% wrong) | −1.17585209122032172461… ✓ | −1.17585… |
| tan(14.13) | off 4e-5 | 293.034591113924226111… ✓ | 293.0345911… |
| asin(0.99999999999) | off 112 LSB | 1.5707918546589416159… ✓ | 1.5707918… |
| acosh(2) | errored | 1.3169578969248167086… ✓ | 1.3169578… |
| tanh(−5.29) | panicked | −0.99994890484746193280… ✓ | −0.9999489… |
| atanh(−0.999…9) | −32.236 + panics | **+**32.5827648921966122309… | **−**32.5827… |

Arm B (compute-tier chain persistence + UGOD) is correct to ~36 digits on every
case Arm A fails — **vindicating the router/UGOD design**. It surfaced one
genuine FASC bug: `atanh` returns the right magnitude with the **wrong sign**
for negative argument.

## Bug inventory

| # | severity | where | bug |
|---|----------|-------|-----|
| 1 | high | Arm A `asinh/acosh/atanh` | storage-tier composition → gross errors (up to ~47%) + panics; `x²±1` overflows i128 |
| 2 | high | Arm A `tan/asin/acos` | div-by-near-zero at poles/boundaries → panics + large precision loss |
| 3 | high | Arm A `tanh` | `exp(2x)` overflows for large \|x\| (should saturate to ±1) → panics |
| 4 | med | Arm A `sinh/cosh` | composed at storage tier; faithful (≤1 LSB) but not correctly rounded |
| 5 | med | Arm B FASC `atanh` | wrong sign for negative argument (magnitude correct) |
| 6 | low | public API | `DecimalFixed` composed methods `.expect()`-panic on valid inputs — a public method must not panic on in-domain input |

## Recommended fixes

1. **Rewrite the Arm A composed transcendentals to compose at the compute tier**
   using the existing `decimal_compute_{mul,add,sub,div,halve,one,...}`
   primitives + `decimal_sqrt/ln/exp` at compute tier, single downscale at the
   end — mirroring the dedicated engines and the FASC path. The primitives all
   exist; this is consistent with current architecture.
   - `asinh`: `sign(x)·ln(|x| + √(x²+1))` at compute tier (stable form kills the
     cancellation that makes the naive `ln(x+√(x²+1))` wrong for x<0).
   - `acosh`: `ln(x + √(x²−1))` at compute tier.
   - `atanh`: `½·ln((1+x)/(1−x))` at compute tier; **fix the sign**.
   - `tanh`: compute tier; **saturate to ±1** for large \|x\| instead of letting
     `exp(2x)` overflow.
   - `tan`/`asin`/`acos`: compute-tier composition with boundary guards
     (`tan` near poles; `asin`/`acos` at \|x\|=1 → return ±π/2 / 0/π directly).
   - `sinh`/`cosh`: move to compute tier for full correct rounding.
2. **Fix the FASC `atanh` sign bug** for negative argument (bug #5).
3. **No public panics**: in-domain inputs must return a value; the compute-tier
   rewrite removes the overflow that triggers the `.expect()` panics.
4. Optional stopgap: route the imperative composed methods through the FASC path
   internally (it is already correct), accepting FASC overhead, until native
   compute-tier composition lands.

## Notes for the reply to decimal-scaled

- gMath's decimal **dedicated** engines (`exp, ln, sqrt, sin, cos, atan`) are
  correctly rounded — verified against decimal-scaled's own adversarial corpus,
  0 LSB across ~9,000 inputs, his data, reproducible.
- The **composed** functions on the imperative surface are not yet — concrete,
  honest, and exactly the kind of gap he hinted at. Fixes are scoped above.
- The half-away-vs-banker's-rounding question is moot for transcendentals (no
  exact ties); it matters for the decimal **arithmetic** path, where a
  `_with(RoundingMode)` (HalfToEven) option is the relevant ask for finance.
