# gMath Roadmap

Current version: **0.4.34**

This document tracks planned work and known gaps. Items are grouped by priority, not by timeline. Nothing here is a promise — this is a working list for a solo-maintained project.

---

## Delivered

### v0.3.0 — Five profiles

| Profile | Storage | Compute | Bytes | Digits | Status |
|---------|---------|---------|-------|--------|--------|
| realtime | Q16.16 (i32) | Q32.32 (i64) | 4 | 4 | validated vs mpmath |
| compact | Q32.32 (i64) | Q64.64 (i128) | 8 | 9 | validated vs mpmath |
| embedded | Q64.64 (i128) | Q128.128 (I256) | 16 | 19 | validated vs mpmath |
| balanced | Q128.128 (I256) | Q256.256 (I512) | 32 | 38 | validated vs mpmath |
| scientific | Q256.256 (I512) | Q512.512 (I1024) | 64 | 77 | validated vs mpmath |

All profiles use true tier N+1 computation. 18 transcendentals, 4 domains (binary, decimal, ternary, symbolic), FASC zero-allocation stack computation, UGOD tiered overflow delegation, BinaryCompute chain persistence.

### v0.3.89 — TQ1.9 standalone module

Dedicated inference module with AVX2 SIMD, rayon row-parallel dispatch, batch matvec. `TQ19Matrix`, `tq19_dot`, `trit_dot`, packed trit operations, `TRIT_DECODE_TABLE`. 38 tests.

### v0.3.90 — Configurable FRAC_BITS, native transcendental dispatch, inference feature gate

- `GMATH_FRAC_BITS` env var for realtime profile (e.g., Q8.24 via `GMATH_FRAC_BITS=24`)
- Native transcendental dispatch: Q16.16/Q32.32 use hardware i128 instead of software I256
- Fixed `ln_q64_64_native` algorithm bug (additive -> multiplicative decomposition)
- `FixedPoint::sincos_wide(i64)` for wide-range RoPE position encoding
- TQ1.9 gated behind `inference` feature (replaces `parallel`)
- Decimal-to-binary rounding fix (round-to-nearest instead of truncation)

### v0.4.0 — Decimal transcendentals, fractal router, direct engine calls, gmath! macro

**Decimal transcendentals (native engines):**
- 5 native engines: exp (4-stage table decomposition), ln (atanh), sqrt (Newton-Raphson), sin/cos (Cody-Waite + Machin pi), atan (half-angle)
- DecimalCompute StackValue variant + full FASC dispatch
- DecimalFixed imperative type: 18 transcendental methods wired to native engines (eliminated binary round-trip)
- 14 ULP validation tests + 24 FASC integration tests + ~9500 mpmath reference points

**Fractal topology router:**
- Shadow-based operand classifier — factors CompactShadow denominator to determine domain exactness
- Compile-time routing table (5.25 KB .rodata, 21 ops x 16 x 16 classes) — O(1) lookup
- Cross-domain coercion in arithmetic — `gmath("0.1") + gmath("255")` routes to Decimal (was Symbolic fallback)
- Tree walker: `route_expression(&LazyExpr) -> OperandClass`, O(N) bottom-up

**Decimal 4-stage exp tables:**
- `exp(x) = exp(k) * exp(d1/10) * exp(d2/100) * exp(d3/1000) * exp(r)` — 71 cached entries
- Cached-table decimal exp path; narrows the binary/decimal gap

**Direct binary engine calls (FASC bypass):**
- FixedPoint: exp/ln/sqrt/sin/cos/atan use `direct_unary` pattern — upscale -> engine -> downscale
- Avoids the FASC pipeline overhead per call (no LazyExpr, no TLS, no StackValue boxing)

**gmath!() compile-time macro:**
- `g_math_macros/` proc-macro crate — zero external deps, `--features macros`
- Pre-parses decimal/integer literals at compile time, emits direct StackValue construction
- Falls back to runtime `gmath()` for fractions, constants, hex

**FRAC_BITS root cause fix:**
- BinaryTier1 mul/div hardcoded `>> 16` — wrong when `GMATH_FRAC_BITS != 16`
- Fixed: uses `frac_config::FRAC_BITS` for q16_16 profile, hardcoded 16 for all others

**Matrix chain persistence, fused ops (also in v0.4.0):**
- `LazyMatrixExpr` — 14-variant enum, operator overloading, recursive evaluator at ComputeMatrix tier
- `DomainMatrix` — StackValue-tagged matrix, 4 domains + cross-domain routing
- Fused transcendental paths — `evaluate_sincos()`, identity short-circuits (`exp(ln(x))` -> x)
- Fused compute-tier ops — `sqrt_sum_sq`, `euclidean_distance`, `softmax`, `rms_norm_factor`, `silu`

**Tensor decompositions (for inference weight/KV-cache compression):**
- `truncated_svd(a, k)` — keep top-k singular values, O(mk + nk) memory vs O(m² + n²)
- `truncated_svd_auto(a, threshold)` — automatic rank selection via singular value threshold
- `tucker_decompose(t, ranks)` — HOSVD: mode-n unfolding → SVD per mode → core tensor
- `cp_decompose(t, rank, max_iter, tol)` — Alternating Least Squares for rank-R canonical polyadic

**Composed transcendental direct bypass:**
- All *infallible* FixedPoint transcendentals bypass FASC entirely (the fallible `try_*` variants still route through FASC — see 0.5.0)
- Composed functions (tan, sinh, asin, etc.) use direct compute-tier arithmetic
- `pow(x, y) = direct_exp(direct_ln(x) * y)` — zero FASC overhead

**Test count: 1375+ across all 5 profiles, 0 failures, 0 warnings.**

### v0.4.1 — Fused `sinhcosh` hyperbolic pair

Independently-rounded `sinh(θ)` and `cosh(θ)` fail to cancel in expressions like `r = cosh(θ)·p + (sinh(θ)/θ)·v`, producing closure drift on hyperbolic-geometry round-trips at low-precision profiles. Fix: expose a fused pair sharing one `(exp(x), exp(-x))` evaluation so the rounding bias is correlated.

**Delivered:**

- `FixedPoint::sinhcosh` / `try_sinhcosh` — imperative binary, mirrors `sincos` / `try_sincos` shape
- `DecimalFixed::sinhcosh` — imperative decimal, native compute-tier path (no binary round-trip)
- `decimal_sinhcosh(ComputeStorage)` kernel in `decimal_fixed::transcendental::exp`
- `sinhcosh_at_compute_tier(ComputeStorage)` in `stack_evaluator::compute`
- `StackEvaluator::evaluate_sinhcosh` — internal, decimal/binary dispatch
- `canonical::evaluate_sinhcosh(&LazyExpr)` — FASC public surface, parity with `evaluate_sincos`

**Benefits:**

- 2 `exp` evaluations instead of 4 (separate `sinh` + `cosh` each call `exp` twice)
- **Correlated rounding**: sinh and cosh derive from the same `(ep, en)` compute-tier pair, so systematic bias cancels in expressions like `cosh(θ)·p + (sinh(θ)/θ)·v`. Per-function ULP is unchanged (still 1 ULP at final downscale), but the *joint* error is tighter.

**Validation:** mpmath-backed tests in `tests/sinhcosh_validation.rs` on realtime (Q16.16) + compact (Q32.32) profiles. Binary identity `cosh²−sinh² = 1` holds to ≤8 ULP at Q16.16 storage, ≤64 ULP at Q32.32 (both dominated by cosh² amplification of input-representation ULP).

**Deferred for future:** Dedicated mpmath-backed `sinhcosh` validation passes on embedded (Q64.64), balanced (Q128.128), and scientific (Q256.256) profiles. The underlying compute-tier kernels are profile-agnostic and already exercised by the existing per-profile `sinh` + `cosh` validation suites; the fused pair's correctness inherits from those. Run dedicated passes if consumer demand emerges for those profiles.

### v0.4.2 — Internal adoption of `try_sinhcosh` in `HyperbolicSpace`

gMath now consumes its own fused primitive internally. `HyperbolicSpace::exp_map` and `HyperbolicSpace::parallel_transport` (in `src/fixed_point/imperative/manifold.rs`) previously called `try_cosh` and `try_sinh` as two independent transcendentals on the same `theta`. Both sites feed the outputs into expressions where correlated rounding matters:

- `exp_map`: `r = cosh(θ)·p + (sinh(θ)/θ)·v`
- `parallel_transport`: `correction = sinh(θ)·p + (cosh(θ)-1)·u`

Swapping to `try_sinhcosh` shares one `(exp(θ), exp(-θ))` pair at compute tier, so the two rounding errors cancel in the downstream sum instead of accumulating. This directly addresses the closure drift hyperbolic-geometry round-trips can show at low-precision profiles.

**API-compatible:** no surface change — both methods retain their `Result<FixedVector, OverflowDetected>` signatures. Downstream consumers get the tighter bound transparently via `cargo update`.

**Regression-tested:** `tests/l1c_l1d_l3a_validation.rs` (24 tests) and `tests/sinhcosh_validation.rs` (8 tests) pass cleanly on realtime and compact profiles.

### v0.4.24 — Decimal arithmetic correctness + contract validation

Built a contract-validation harness (`tests/decimal_contract_validation.rs`)
grading gMath's decimal domain against mootable/decimal-scaled's independent
mpmath corpus (pinned), under gMath's own declared rounding rule. It surfaced —
and this release root-causes — three bug classes:

- **Composed decimal transcendentals** (tan, asin, acos, sinh, cosh, tanh,
  asinh, acosh, atanh) composed at storage tier with raw i128 arithmetic,
  overflowing for large operands and panicking. Rewritten to compose at the
  compute tier with a single downscale — now correctly rounded.
- **Parser sign drop**: `gmath_parse` lost the sign of negative decimals with a
  zero integer part (`-0.5` parsed as `+0.5`), via `integer_part < 0` being
  false for `-0`. Fixed at the source.
- **Decimal mul/div overflow**: `divmod_d256_by_i128` overflowed its u128
  remainder for divisors > 2^64, and `banker_round_decimal_i128` mis-rounded odd
  divisors and exact scale-0 results. Both fixed; mul/div rewritten UGOD-shaped
  (narrow i128 tier → widen to a 256-bit intermediate on overflow → saturate,
  never panic), banker's rounding preserved.

Both user-reachable decimal paths — imperative `DecimalFixed` and canonical
`gmath()`/FASC — grade **0 LSB** across all 15 transcendentals, atan2, and
add/sub/mul/div over decimal-scaled's d18+d38 tiers and all scales. Audited
every tier×op across decimal/binary/ternary: the bug class was contained to the
imperative `DecimalFixed` path (the tiered arithmetic already widened correctly).
CI (`.github/workflows/decimal-precision.yml`) gates the harness plus
multi-profile lib tests on every push.

### v0.4.25 — Trit-plane weight formats + fused attention hardening

`PlanarTQ19` (per-digit ternary planes, density-chosen encodings) and
`HybridTQ19` (12-bit packed low part + sparse high corrections) — lossless
re-encodings of `TQ19Matrix` with bit-identical matvec at 25–29% fewer weight
bytes. `fused::softmax_mix` numerator/exp-sum accumulate with overflow
detection. CI: `fused-tq19-precision` workflow gates fused ops and TQ1.9
exactness across all five profiles plus a Q22.10 floor job.

### v0.4.26–v0.4.27 — U1 fused kernels + try_* domain-error contract

`fused::{euclidean_distance_squared, dot, mobius_denominator_sq}` (U1 consumer
asks: squared-space scoring and Möbius-ratio kernels without the sqrt
round-trip). `try_ln`/`try_sqrt` again return `DomainError` for out-of-domain
inputs — the v0.4.0 direct-engine switch had bypassed the FASC domain checks
and misreported them as `TierOverflow`.

### v0.4.28 — round_to_storage overflow-panic

The shared fused/linalg downscale silently wrapped results exceeding the
storage tier (a squared distance of 520,000 returned as −4288 on Q16.16). Now
panics like every other infallible downscale. Fused test suite made
multi-profile compliant (representable tolerances, narrow-profile-safe data).

### v0.4.29 — Saturating exp downscale + ceiling guards

`downscale_q64_to_q32` wrapped oversized Q64.64 results (including the exp
overflow sentinel) via a plain cast: `fused::silu(x)` for x ≲ −30 returned the
input unsquashed, `tanh(25)` returned −1. Now saturates; every exp consumer
whose follow-up add could wrap on a saturated exp is guarded (silu → 0,
tanh → 1, cosh → loud overflow, imperative + FASC paths). Property test pins
exp monotonicity over the full storage range.

### v0.4.30 — RowScaledTQ19 (TQ1.9-R)

Per-row quantization scales (`s_rel` in unsigned Q32.32 relative to the global
step): adapts the step to each row's max, ~20× matvec output error reduction
on BF16-trained MoE weight distributions at unchanged 2 bytes/weight. Gated to
q16_16/q32_32. Review hardening: loud range check on the scale multiply,
independent i128-oracle test.

### v0.4.31 — Wide-output matvec_q2f family

`matvec_q2f`/`_par`/`_batch_par` on all four TQ1.9 forms plus `tq19_dot_q2f`:
the exact row accumulator at 2·FRAC_BITS fractional precision with exactly one
rounding, for consumers whose signal sits below the storage rounding floor
(fine-grained-MoE expert outputs). Narrowing contract pinned by property
tests: `q2f / (1 << FRAC_BITS)` reproduces the narrow matvec bit-for-bit for
the three unscaled forms; `RowScaledTQ19` applies the scale to the wide dot
(±1 storage LSB vs the narrow path for non-unit scales, exact at unit scale).

---

### v0.4.32 — Public compute-tier transcendentals

`g_math::compute_tier` (feature `inference`): the tier-N+1 engines exposed
directly over raw `ComputeStorage` at 2·FRAC_BITS precision — `exp`, `ln`,
`sqrt`, `sinhcosh` plus `sigmoid`/`softplus`/`ln1p` stable compositions,
with `from_fixed`/`to_fixed`/`try_to_fixed` conversions. Same engines as
every other path (path independence pinned by test); format-compatible
with the wide-output `matvec_q2f` accumulators. exp saturates at
`ceiling()` and never wraps; domain violations panic; storage conversions
fail loud. mpmath-gated at q16_16/q32_32 with measured tolerances —
storage level exact (0 LSB) for all functions at both profiles. Closes
the consumer ask that had downstream re-deriving integer-only wide-tier
exp/sigmoid/softplus/ln1p.

---

### v0.4.33 — Balanced ternary contract + validation; inv_sqrt family

**DELIVERED 2026-08-11.** The balanced-ternary domain is now formally
defended rather than merely supported (contract + oracle + exhaustive +
theorem suites + `ternary-domain` CI), two latent ternary defects found by
the suite are fixed (`-0.x` sign loss in `from_str`; silent wrap in
`ternary_to_storage` on narrow profiles — now loud `TierOverflow`), and the
consumer-requested reciprocal-norm family shipped additively:
`FixedPoint::inv_sqrt`/`try_inv_sqrt` and `fused::inv_sqrt_sum_sq`
(compute-tier throughout, one rounding; one inv_sqrt + N multiplies
replaces N per-component divisions in normalization). Original scope
below for reference:

- `docs/design/BALANCED_TERNARY_CONTRACT.md` — representation invariants,
  canonicalization, and the **tie-free rounding theorem**: for values with a
  finite balanced-ternary representation, truncating trits is round-to-nearest
  with error strictly below ulp/2 (discarded tail ≤ ½·(1−3⁻ᵐ)·ulp), so
  ternary-closed add/sub/mul need no tie-breaking logic and satisfy
  `round(−x) = −round(x)` by construction. The contract also pins the
  boundary cases where ties *do* reappear (conversion into ternary — binary ½
  is `0.111…` repeating — and division) and specifies the rule used there.
  Sibling invariant: ranges are symmetric `[−(3ⁿ−1)/2, +(3ⁿ−1)/2]`, so
  negation can never overflow (no two's-complement MIN edge case).
- Reference trit-vector oracle (test-only, deliberately simple `Vec<i8>`).
- Exhaustive small-width add/sub/mul vs the oracle; boundary families
  (±3ⁿ, ±3ⁿ±1, long runs of ±1 trits, alternating patterns).
- Theorem tests: tie-free truncation, negation symmetry/no-overflow, the
  conversion-tie case pinned as documented behavior.
- UGOD promotion at 3ⁿ tier boundaries; canonical↔imperative path
  equivalence; ternary↔binary/decimal coercion.
- CI job (no table rebuild needed — cheap on every push).

### v0.4.34 — Ternary routing column (Add/Sub)

**DELIVERED 2026-08-14.** `DomainChoice::Ternary` + table column for
cross-domain Add/Sub of 3-adic operands (exact by construction), silent
fallback when coercion overflows narrow storage, classifier now reads
Symbolic denominators directly, and the unchecked-cast wrap in
`convert_to_ternary` fixed. Mul/Div deliberately excluded — products can
leave the tier's exactness range where ternary truncates but symbolic is
exact, and the class mask cannot see denominator exponents; a
dispatch-time shadow-exponent guard is the future path. Reasoning:
`docs/design/TERNARY_ROUTING_COLUMN.md`. Measured end-to-end: ~1.06×
vs the rational fallback (parse-dominated); the value is architectural.

---

## Next: 0.5.0 — Correctness audit + remaining composed transcendental bypass

**Status: staged on `main`, not yet released.** Items 0 through 2c below are
delivered, swept green on all five profiles, and CI-green; item 3 is deferred
past the cut. What remains for the release itself: version bump, CHANGELOG
dating, and the consumer notice for the two bit-level changes (binary
mul/div rounding, ternary raw scales). Until the tag lands, the published
crate is 0.4.34 and `Cargo.toml` still reads 0.4.34 by design.

### 0. Narrow-profile integer-literal parse fallback — DELIVERED 2026-08-14

**Fixed on main** (post-0.4.34): oversized integer literals now fall back
to the Symbolic domain in `parse_integer` (all three narrow-profile arms);
regression test `oversized_integer_literal_falls_back_to_symbolic` runs on
every profile, and the router/domain integration suites now run on
realtime+compact in CI (ternary-domain workflow) so this class cannot hide
again. Original finding:
On realtime (Q16.16), any integer literal beyond the binary storage range
(`32768`+) fails PARSE with `Overflow` instead of falling back to the
symbolic domain — so `evaluate(gmath("1000000") * gmath("0.001"))` errors
on realtime while succeeding everywhere else. UGOD's contract says the
symbolic ladder top never fails; the parse path should route oversized
integers to Symbolic. Pre-existing at least since 0.4.32 (bisected);
invisible because no CI job runs the router integration suite on realtime
(`large_integer_times_decimal` fails there today). Fix parse routing, make
the test profile-aware, and add a realtime router-integration CI job.

### 0b. Unsigned widening-multiply call-site audit — DELIVERED 2026-08-23

Every call site of the unsigned family (`mul_to_i512/i1024/i2048`)
enumerated and classified (appendix in `docs/design/ROUNDING_CENSUS.md`):

- **Sign-wrapped (correct)**: linalg dot/product helpers, decimal compute
  engine, compute_multiply, sincos/atan via the sign-safe
  `multiply_i1024_q512_512`, the 0.4.34-era ternary fixes.
- **Positive-by-construction (now debug_assert-ENFORCED)**: exp table
  chains + Taylor, ln Taylor remainder, sqrt Newton — every such site
  carries a non-negativity assert, so every debug/test run polices the
  invariant instead of trusting comments.
- **De-shadowed**: exp and sqrt each had a private UNSIGNED
  `multiply_i1024_q512_512` shadowing the sign-safe pub(crate) one —
  renamed `*_nonneg` with asserts; the shadowing hazard is gone.
- **Latent bugs fixed in dead code**: the three pow `y·ln_x` sites were
  genuinely sign-broken but `pow_tier_n_plus_1.rs` has ZERO external
  callers (FASC/imperative compose pow as exp(y·ln x) through the safe
  path) — sign-wrapped anyway. RESOLVED 2026-08-23: module REMOVED per
  owner decision (with its direct-engine tests and reference data);
  composed pow keeps its 0-ULP gates.
- **Modular-truncating wide `Mul` fallbacks** (I512/I1024 schoolbook mod
  2^N) are sign-correct by modular arithmetic; exercised with negatives
  by the 0.4.34 tier-5 lattice tests.
- New gate: `tests/negative_operand_battery.rs` — odd/even transcendental
  symmetries bit-exact, negative-intermediate chains (exp∘ln on x<1,
  sinh/cosh of −x), negative FASC compute products; all profiles.

### 0c. Uniform rounding policy — DELIVERED on main 2026-08-23

**Implemented exactly as analyzed below**; permanent gate
`tests/rounding_unification.rs`; per-site evidence and implementation
corrections in `docs/design/ROUNDING_CENSUS.md`; CONTRACT.md §3 collapsed
to one row per domain. Original analysis retained:

**Owner-directed 2026-08-14** ("having different rounding makes it a more
difficult library to work with"). Today rounding differs not just per
domain but per PATH within a domain (CONTRACT.md §3): binary multiply is
banker's imperatively but ties-up canonically (1 ULP apart on exact ties,
measured); DecimalFixed is banker's while the canonical decimal divide
truncates; ternary truncates toward zero despite nearest being provably
tie-free.

**Analysis of the three options:**

- *No rounding (exactness preservation)* is not an alternative — it is
  deferral. It is superior wherever a result stays representable (the
  canonical decimal multiply already grows decimal places instead of
  rounding, and tier N+1 defers all compound rounding to one downscale),
  but every domain eventually hits its scale cap and needs a rule. So the
  policy is: **exact when representable, round once otherwise** — with
  the remaining question being which tie rule fires on that one rounding.
- *One global rule* is simplest to state but fights both convention and
  the codebase: banker's everywhere would rewrite the wide-tier downscale
  that every 0-ULP validation is pinned to (huge blast radius, zero
  consumer benefit — compound paths never tie); ties-up everywhere would
  break the banker's behavior financial consumers expect of DecimalFixed.
- *One rule per domain, uniform across paths* matches convention AND
  minimizes result churn. **Recommended:**

| Domain | Rule everywhere | Rationale | What changes |
|---|---|---|---|
| Binary | nearest, ties toward +INF | the wide-tier downscale — the only rounding compound results ever see — is already ties-up everywhere; aligning the imperative mul kernel is a one-branch change | `multiply_binary_i128` (+ AVX2 twin): banker's -> ties-up; closes the measured 1-ULP path divergence |
| Decimal | banker's (half-even) | the accounting convention, and what DecimalFixed already does on mul AND div | canonical decimal divide: truncation -> banker's; dp-cap rounding aligned |
| Ternary | nearest (tie-free) | 3^m is odd — nearest can never tie (contract theorem); no rule needed at all | mul/div/div3: toward-zero (<1 ulp) -> nearest (<0.5 ulp); resolves the 0.4.33-flagged decision; div3 becomes a true trit shift |

End state a consumer can memorize in one sentence: *binary rounds to
nearest ties-up, decimal rounds banker's, ternary rounds to nearest
(which cannot tie)* — identical on every path within each domain.

All three alignments change results only on exact ties or by <1 ulp on
currently-truncated ops: breaking-precision class, so they ship together
in 0.5.0 with a consumer notice (consumers freezing hashed outputs should
pin versions across the boundary). Add tie-case and cross-path
equivalence tests per domain to the path-independence suite.

### 0d. "Realtime cosine plateau" — SOLVED 2026-08-23 (was a materialization bug)

Investigation cleared the kernel entirely: the q16_16 sin/cos engines are
correct (cos(0.1) = 0.99524 at the kernel, imperative path bit-perfect).
The real defect: DecimalCompute results materialized at a fixed
`DECIMAL_STORAGE_MAX_DP - 2` in THREE places (Display, to_decimal_string,
to_rational) — imperceptible slack on wide profiles, but HALF of
realtime's four decimal digits, so 0.9952 rendered as "1.00" and every
display-round-tripping test saw a "plateau". Fixed with adaptive
materialization: try full MAX_DP, step down only when the magnitude
genuinely needs fewer decimals (checked downscale, never a wrap). Wide
profiles gained their 2 withheld digits back as a side effect. The
0d-era test-tolerance workarounds were REVERTED — original strict
tolerances pass again on realtime. The related FASC exp(20) realtime
TierOverflow was resolved under item 1: it is correct-and-loud (e^20
exceeds every realtime representation), not a missing fallback.

### 1. UGOD multi-tier promotion verification — DELIVERED 2026-08-23

**Verdict:** the original concern (promoting to symbolic after a single
tier) was NOT the defect — mid-ladder promotion (binary tiers 1→4) was
always correct (multiply/divide chain tiers explicitly; add/sub's
single-step promotion is provably sufficient since a sum grows by at
most one bit). The TOP of every ladder was broken instead, and wrapped
SILENTLY rather than falling back:

- Binary Tier-4/5 mul truncated wide products unchecked (balanced
  `1e20 × 1e20` → 1.318e38 wrap garbage); Tier-4/5/6 add/sub used bare
  wrapping operators (embedded `9e18 + 9e18` → 0.0). Now checked, with
  4→5→6 promotion arms.
- `binary_to_storage` narrowed promoted raws with bare casts; now
  fits-checked, and the FASC binary arms fall back to the exact
  rational path on `TierOverflow` — the true ladder top: exact or loud.
- Divide mislabeled quotient overflow as `DivisionByZero` (zero check
  now decided once at ladder entry).
- Symbolic ladder: try_multiply's "promotion" retried at the same i128
  width (never reached Massive/I256); `divide_mixed_tiers` didn't
  escalate although a quotient can need a wider tier than either
  operand (9e18 ÷ 1e-9 = 9e27). Both now climb.
- FASC symbolic/ternary→binary coercion shifted i128 before any range
  check — symbolic 1e20 coerced on embedded wrapped mod 2^64 into a
  PLAUSIBLE WRONG value. Now checked nearest-ties-+∞ at tier N+1 width.
- Two adjacent finds: narrow-profile fractional literals beyond the dp
  cap silently parsed to truncated values (realtime `"0.000000001"` →
  exactly 0) — now Symbolic-fallback like item 0; and the scientific
  formatter displayed integer parts mod 2^128 (i128 squeeze) — now
  digit-at-a-time at I256 width.

Gate: `tests/ugod_promotion_validation.rs` (all 5 profiles green; in
ternary-domain CI on realtime+compact). The 0d-flagged realtime FASC
exp(20) TierOverflow is RESOLVED as correct-and-loud: e^20 ≈ 4.85e8
exceeds every realtime representation (Q16.16 max ≈ 3.3e4), so a loud
TierOverflow is the contract — not a missing fallback.

### 2. Complete FixedPoint direct-engine bypass for composed transcendentals — DELIVERED 2026-08-23

The fallible `try_*` variants (tan/atan/asin/acos/sinh/cosh/tanh/asinh/
acosh/atanh) no longer route through FASC (`try_apply_unary(LazyExpr)`);
each is now a direct compute-tier composition mirroring its infallible
twin — same engines, same formula, same single downscale. Error contract
preserved: `DomainError` on domain violations (0.4.27), `TierOverflow`
on storage overflow, `asin(±1) = ±π/2` boundary shortcut, tanh ceiling
saturation to exactly 1. Gate: `tests/try_direct_bypass_validation.rs` —
bit-identity with the infallible twins on in-domain inputs (which are
0-ULP gated), typed domain errors, loud overflow.

### 2b. Ternary tier resize — DELIVERED 2026-08-23 (owner-approved)

Every balanced-ternary tier now fills its storage word: TQ10.10 (i32),
TQ20.20 (i64), TQ40.40 (i128), TQ80.80 (I256), TQ160.160 (I512),
TQ320.320 (I1024) — +25% trits per tier, same words, same speed, clean
2× ladder. BREAKING for persisted ternary raws (all scale factors
changed; tie pins now 0.5 → 29,525 / −0.5 → −29,524). The FASC
binary→ternary coercion's tier-3 arm moved to I256 intermediates (the
larger 3^40 scale overflowed the i128 path). TQ1.9 unaffected. Full
capacity math: docs/design/BALANCED_TERNARY_CONTRACT.md §1b. Resolves
the 0.4.33-flagged "tier resize" owner decision — batched with the 0c
rounding change so consumers absorb ONE bit-compatibility boundary.

### 2c. Dead pow engine removed — DELIVERED 2026-08-23 (owner-approved)

`pow_tier_n_plus_1.rs` deleted with its direct-engine tests and
reference data: zero production callers since the composed
exp(y·ln x) path shipped; its latent sign bugs (0b audit) are moot.
Composed pow keeps its 0-ULP mpmath gates.

### 3. Stack evaluator `profile_dispatch!` macro

**Priority:** MEDIUM — reduces maintenance burden, prevents cfg copy-paste bugs
**Effort:** ~750 lines, 2 sessions

Extract macro to replace the 5-way `#[cfg(table_format)]` blocks across all stack evaluator submodules. Each profile-conditional function currently has 5 copy-paste arms. A macro reduces this to 1 invocation per function and makes future profile additions mechanical.

---

## Requested by consumers (gHyper/gFile) — U2: cross-profile rounding contract

Different profile bit-widths necessarily round to different values (a Q32.32
result cannot hold Q64.64 digits) — that divergence is inherent and out of
scope. What IS in scope is stating and CI-testing the contract BETWEEN
profiles:

1. **Widening is exact**: every narrower-profile value embeds losslessly in
   any wider profile (pure bit-shift; test it).
2. **Narrowing is a single correct rounding**: because transcendentals are
   computed at the compute tier and rounded once to storage, a narrow
   profile's result should equal round_narrow(exact) — i.e. no
   double-rounding artifacts (round_32(round_64(x)) != round_32(x) edge
   cases). State it, and CI-gate it with cross-profile fixture comparisons.
3. **Serialization semantics for foreign profiles**: `profile_tag()` exists;
   document whether a reader encountering a foreign-profile raw value must
   reject or may convert (widening ok, narrowing rounds — and say which
   rounding mode).

Context: gFile's `.htt` format pins Q64.64 and byte-determinism is CI-gated
per platform; this contract is what would make any future cross-profile
tooling (e.g. compact-profile analytics over embedded-written data) sound
rather than accidental.

## Future — High Priority

### Complete the imperative (non-routed) layer for every domain

**Priority:** HIGH — the largest ergonomics gap in the public surface

The imperative layer (`FixedPoint`, `FixedVector`, `FixedMatrix`, and everything
built on them — decompositions, matrix functions, manifolds, Lie groups, ODE,
tensors) is the direct, no-routing, `Copy`-semantics path: one call is one engine
invocation, no lazy tree, no fractal-router dispatch. Today it exists **only for
the binary domain**. `DecimalFixed` covers decimal *scalars* with native
transcendentals, but there is no `DecimalVector`/`DecimalMatrix`, and balanced
ternary and symbolic rational have no imperative vector/matrix surface at all.

The goal is parity: give each domain gMath already supports (decimal, balanced
ternary, symbolic rational) the same direct imperative surface binary has, so a
consumer who knows their domain can skip the router overhead in *any* domain, not
just binary. Concretely:

- `DecimalVector` / `DecimalMatrix` over `DecimalFixed<N>`, with compute-tier dot
  accumulation mirroring `FixedVector`/`FixedMatrix`.
- An imperative surface for balanced ternary vectors/matrices (beyond the current
  TQ1.9 inference path).
- Symbolic rational vectors/matrices for exact linear algebra.
- Where a domain lacks native transcendentals (ternary, symbolic), document the
  binary-compute bridge explicitly rather than silently routing.

This is a multi-session track; sequence decimal first (highest consumer demand),
then ternary, then symbolic. Relates to *Symbolic rational transcendentals* and
*Imperative geometry methods* below.

### Batch/vectorized API

SIMD-friendly array processing for FixedPoint operations beyond TQ1.9. Bulk exp, sqrt, and arithmetic over vectors would accelerate softmax, RMSNorm, and embedding decode. Compelling with Q32.32 (8x i32 per AVX2 register) and Q16.16/Q8.24 (16x i16).

### Ternary test coverage

Balanced ternary arithmetic lacks a dedicated validation suite. The domain works but needs stress-testing against reference values. Low effort, fills a known quality gap. **Now scheduled: see "Planned: 0.4.33" above.**

### Interval arithmetic — certified enclosures

A first-class interval type ([lo, hi] guaranteed to enclose the true value) built
on the existing tier-N+1 machinery. The downscale step already rounds once;
adding round-toward-−∞ and round-toward-+∞ modes beside the current
round-to-nearest is a branch on the discarded bits, not new math — that directed
(outward) rounding is what makes enclosures sound. With gMath's correctly-rounded
transcendentals, interval versions come nearly for free: evaluate f at the
endpoints, widen by the known ≤1-ULP rounding error. Turns "correctly rounded"
into "certified bound" — the error-transparency story for validation/anomaly
consumers (e.g. interval-certified scores: d² ∈ [lo, hi], "certainly outside" vs
"within numerical noise"). Real work is the dependency problem (centered/affine
arithmetic to curb interval widening) and non-monotonic extrema; ship a
monotonic-first v1. Reference: Moore/Kearfott; IEEE 1788-2015 (decorations).

### Exact geometric predicates

Certified orient2d/orient3d, incircle/insphere, segment intersection, and
containment, built on the exact rational (BigInt a/b) and integer fixed-point
domains. Each predicate is the sign of a determinant polynomial in the inputs,
and only the sign matters — so exact arithmetic makes the verdict provably
correct, including the exact-zero degenerate cases (collinear/cocircular) that
floating point cannot decide reliably. Speed via the standard filter→fallback
pattern: the interval type (above) resolves the common case, exact arithmetic
runs only near zero. The same sign-of-determinant primitive certifies matrix
positive-definiteness/rank (a real need for any SPD/Cholesky consumer — replaces
blind diagonal regularization). Enables a topological "shape over a point cloud"
layer (Delaunay / alpha-complex / persistent homology — integer reduction, zero
float) in low dimension. Provenance: Delone (1934, "Sur la sphère vide") +
Voronoy; Shewchuk adaptive predicates; CGAL exact-computation paradigm.

### Imperative geometry methods — UGOD + FASC integration

Upstream `square()`, `reciprocal()`, `powi()`, `manhattan_distance()`, `mul_vector()` etc. as first-class UGOD-dispatched, FASC-computed methods. ~800 lines.

### Symbolic rational transcendentals

The third compute domain after Binary and Decimal. Transcendental chains where ALL intermediates are exact rational numbers (num/den BigInt pairs). Unbounded precision — no ULP concept, just exact arithmetic until final materialization. Ultra-precision mode for scientific computing and formal verification.

### n-D Clifford algebra with Vahlen matrices (L4B) — 0.5.0

Cl(n,0,1) Clifford algebra with Vahlen matrix representation. Multivector arithmetic, geometric product, inner/outer products, grade extraction, and Vahlen matrix Mobius transformations. Fills a gap I'm not aware of an existing zero-float Rust crate addressing. Targeted for v0.5.0 release.

### Custom FRAC_BITS for non-realtime profiles

Extending `GMATH_FRAC_BITS` to compact (i64), embedded (i128), and higher profiles. Requires verifying that `COMPUTE_FRAC_BITS = 2 * FRAC_BITS` does not exceed the native transcendental tier's capacity. Deferred until demand.

### Public API stabilization

Pre-1.0 audit of exports, feature gating, StackValue extraction methods. Five API tiers documented (FASC, Imperative, Fused, Geometric, TQ1.9) but export hygiene not yet audited.

### Additional decimal operations

decimal-scaled's corpus covers several functions gMath doesn't expose in the
decimal domain yet, all composable from the existing ln/exp/sqrt compute-tier
engines: `powf` (xʸ — financially core, already present on binary `FixedPoint`),
`rem` (modulo), and the `log2` / `log10` / `exp2` / `log(base)` family (trivial
ln/exp compositions). `cbrt` and `hypot` are lower priority. Implement the
subset that serves real use cases (finance: `powf`/`rem`; ML: `log2`) — not for
corpus parity.

### Decimal sin/cos 3-stage tables

Precomputed sin/cos tables for the decimal engine. Deferred — analysis showed Taylor computation is a small fraction of FASC pipeline time, so tables give little gain through the full pipeline. Revisit when decimal sin/cos becomes a bottleneck (currently not — binary sin/cos handles most inference).

---

## Non-goals

- **Floating-point interop beyond convenience**: `to_f64()`/`from_f64()` exist for user convenience. Internal float usage is architecturally forbidden.
- **Dynamic precision selection**: Profiles are compile-time. Runtime tier selection within a profile is handled by UGOD, but the base storage tier is fixed at build time.
- **GPU compute**: The library targets CPU determinism. GPU offload would compromise the cross-platform bit-identical guarantee.
