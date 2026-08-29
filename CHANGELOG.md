# Changelog

All notable changes to gMath will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Certified interval arithmetic**: `g_math::fixed_point::Interval`
  (module `imperative::interval`). An enclosure `[lo, hi]` that is sound by
  construction: every product of storage values is formed exactly at the
  compute tier, and every narrowing back to storage rounds the lower endpoint
  toward negative infinity and the upper toward positive infinity.
  Operations: `+ - * /` and `Neg` (with `try_*` twins), `sqrt` (certified a
  posteriori by the exact integer check `k^2 <= n < (k+1)^2`, so the
  certificate does not depend on the engine), `dot` and `quadratic_form`
  (exact accumulation, one narrowing per stage), and the certainty predicates
  `contains`, `contains_zero`, `is_certainly_positive`, `is_certainly_negative`,
  `width`, `is_point`. Endpoint arithmetic never wraps: storage overflow is a
  typed `TierOverflow`, and dividing by an interval containing zero is
  `DivisionByZero`. No transcendental is provided; their accuracy is measured
  at test points rather than proven over the domain, and an interval widened
  by a measured error would not be an enclosure. Design and measurements in
  `docs/design/CERTIFIED_INTERVALS.md`. Measured at Q64.64 on a 23-dimensional
  quadratic form: 5 to 64 ulp wide on values of order 10, zero enclosure
  failures, unmoved by ill-conditioning, at 106 to 113 percent of the scalar
  path.
- Directed narrowing points `downscale_to_storage_floor` and
  `downscale_to_storage_ceil` beside the existing nearest variant, crate
  internal. The scalar paths are untouched; one rounding rule per domain
  still holds on every scalar path.
- `tests/interval_enclosure.rs`: the permanent gate (directed rounding
  brackets the nearest scalar by at most one ulp including negatives and
  constructed ties, enclosure against exact integer references, the sqrt
  certificate on sampled inputs, typed errors), on every profile and in the
  narrow-profile CI matrix.

## [0.5.0] - 2026-08-24

### ⚠ Please read before upgrading to 0.5.0

0.5.0 makes gMath's answers more accurate. The side effect: some results
now end in a different final digit than 0.4.x gave you. Nothing became
less precise: the new values are simply closer to the true answer.

**Do you need to act?**

- **You compute and display results** → No. You just get better answers.
- **You save results, hash them, or compare them against numbers an older
  version produced** → Yes. Old and new values won't always match on the
  last digit. Pick one version per dataset and regenerate in one go
  rather than mixing.
- **You need the same input to give the same answer on every machine** →
  Nothing changes. That guarantee is untouched.

**What changed, in plain terms**

1. **Multiply and divide now round to the nearest value.** Some of them
   used to just drop the leftover instead, and which way they dropped it
   depended on which profile you built. Roughly half of all
   multiplications and divisions that don't come out exact will end in a
   different last digit.
2. **Balanced-ternary values sit on a finer grid.** Each ternary tier now
   fits 25% more digits into the same amount of memory. The values mean
   the same thing, but the raw stored numbers are different: the same
   measurement written on a finer ruler. Ternary weights for inference
   (TQ1.9) are not affected.

**The rounding rules now, one per domain.** The same rule applies on
every path (direct calls, expressions, fused ops) and every profile:

| Domain | Rule |
| ------ | ---- |
| Binary | Nearest; a value exactly halfway goes up |
| Decimal | Exact when the result fits, otherwise banker's rounding (halfway goes to the even digit) |
| Balanced ternary | Nearest; multiplication can never land exactly halfway, and where halfway is possible it goes up |

One deliberate exception: the TQ1.9 `matvec_q2f` inference path still
truncates, because its published contract guarantees it reproduces the
narrow matvec bit for bit.

**The rest**: `exp`, `ln`, `sqrt`, `sin`, `cos`, every other
transcendental, and chained expressions return the same bits as 0.4.34;
their reference tests are unchanged and still pass. It is plain
arithmetic, and matrix work built on it, that can move by one digit.

If in doubt: stay on 0.4.34 and upgrade when you can regenerate stored
data in one pass.

---

### Changed: 0.5.0 rounding unification (breaking-precision)

One rounding rule per domain, identical on every path (imperative,
canonical/UGOD, fused, and coercions), replacing rules that differed by
operation, by path, and (imperatively) by profile. Gated permanently by
`tests/rounding_unification.rs` (cross-path bit-equality sweeps plus
constructed exact-tie inputs, per profile). All five profiles green,
including the scientific 18/18 transcendental 0-ULP gate.

- **Binary: nearest, ties toward +∞, everywhere.** Imperative multiply
  was floor (realtime/compact), banker's (embedded), truncation
  (balanced/scientific); imperative divide truncated on all profiles;
  UGOD tier divide was half-away. All now match the wide-downscale rule.
  This REPAIRS path independence for plain mul/div (measured divergence
  before: 48.7% of sampled products, 1 ulp).
- **Decimal: exact when representable; banker's where rounding occurs.**
  Canonical divide tiers 1–5 discovered to be exact-or-rational-fallback
  (kept: better than rounding); the tier-6 best-effort arm moved from
  truncation to banker's; `decimal_to_binary_storage` coercion unified
  from per-profile truncation/add-half to nearest ties-+∞.
- **Ternary: nearest.** Multiply and `div3` are tie-free (odd scale:
  the 0.4.33 contract theorem, now shipped: error halves to ≤ ½ ulp and
  `div3` becomes a true trit shift); divide and conversion-in round
  nearest with ties toward +∞, a documented tie asymmetry, with the sign
  threaded through `from_str`. (The pins moved with the tier resize
  below: `0.5` → raw 29525, `-0.5` → raw −29524.)
- Contracted exception: TQ1.9 `matvec_q2f` narrowing stays truncation
  per its published 0.4.31 bit-reproducibility contract.
- Consumer notice: results may move by up to 1 ulp on direct storage-tier
  multiplies/divides (an accuracy improvement: nearest ≥ floor/trunc);
  consumers freezing hashed or persisted outputs should pin versions
  across this boundary. Compound/tier-N+1 results are unchanged.

### Audited: unsigned widening-multiply call sites (0.5.0 item 0b)

- Every `mul_to_i512/i1024/i2048` call site enumerated and classified;
  positive-by-construction sites (exp/ln/sqrt internals) now carry
  debug_asserts so the invariant is machine-checked in every test build;
  two private UNSIGNED helpers that shadowed the sign-safe
  `multiply_i1024_q512_512` renamed `*_nonneg` (hazard removed); three
  genuinely sign-broken `y·ln(x)` multiplies fixed in
  `pow_tier_n_plus_1.rs`: dead code with zero external callers (pow is
  composed as exp(y·ln x) through the sign-safe path), and the module was
  removed later in this same release (see Removed, below). New `negative_operand_battery` test
  (odd/even symmetries bit-exact, negative-intermediate chains) on every
  profile.

### Changed: ternary tier resize: TQ10.10 … TQ320.320 (breaking, owner-approved)

Every balanced-ternary UGOD tier gains 25% more trits in the SAME storage
word: TQ8.8→TQ10.10 (i32), TQ16.16→TQ20.20 (i64), TQ32.32→TQ40.40
(i128), TQ64.64→TQ80.80 (I256), TQ128.128→TQ160.160 (I512),
TQ256.256→TQ320.320 (I1024). A trit carries log2(3) ≈ 1.585 bits; the
old counts (chosen to mirror the binary tier names) left ~20% of every
word unused. The new counts fill the words; same storage, same
instruction count, 25% more ternary precision digits, and a clean 2×
ladder (10→20→40→80→160→320).

- BREAKING for persisted/hashed ternary raws: every scale factor changed
  (Tier 1 scale is now 3^10 = 59,049). Pinned conversion constants moved
  with it: `0t0.5` → raw 29,525, `-0.5` → −29,524 (tie asymmetry
  preserved), Tier-1 window ±29,524.
- Public fn renames follow the formats (`multiply_ternary_tq10_10` etc.);
  `SCALE_TQ10_10/TQ20_20/TQ40_40` replace the old constants.
- The FASC binary→ternary coercion's tier-3 arm now runs its
  multiply-divide at I256 width: with the larger 3^40 scale, a binary
  raw numerator times the scale can exceed i128 (caught by the mode-
  routing suite: `binary:ternary` sin(1.0) overflowed the old i128 path).
- The same coercion now targets the PROFILE's own tier and scale
  (realtime tier 1 / 3^10, compact tier 2 / 3^20): the old code borrowed
  the tier-3 arm on narrow profiles, whose 3^40-scaled raws no longer
  fit their storage (routed `0t2 + 1/3` on compact went TierOverflow
  instead of exact).
- `from_tier_raw`'s tier-1/2 constructors used bare `as i32`/`as i64`
  casts: oversized raws silently wrapped; now checked `TierOverflow`
  (wrap-defect class).
- `from_str`'s Tier-3 fractional multiply is now checked: long
  fractions cascade to Tier 4 instead of overflowing (latent pre-resize
  hazard, window merely shifted by the resize).
- TQ1.9 (`g_math::tq19`, the packed inference weight format) is a
  separate type and is NOT affected.
- Rationale and word-capacity math: docs/design/BALANCED_TERNARY_CONTRACT.md §1b.

### Removed: dead `pow_tier_n_plus_1.rs` engine (owner-approved)

The dedicated pow engine was superseded before it was ever wired in:
shipping pow (FASC and imperative) composes exp(y·ln x) at the compute
tier, which the 0-ULP suites validate. The module had zero production
callers; its three latent sign bugs were found and fixed in the 0b audit
and are now moot. Its direct-engine tests and reference data
(`POW_REFS`, generator section) are removed with it; the composed pow
keeps its own mpmath 0-ULP gates (integer and fractional exponents) in
`fasc_ulp_validation`.

### Changed: fallible composed transcendentals bypass FASC (0.5.0 item 2)

The `try_*` variants of the composed transcendentals (`try_tan`,
`try_atan`, `try_asin`, `try_acos`, `try_sinh`, `try_cosh`, `try_tanh`,
`try_asinh`, `try_acosh`, `try_atanh`) are now direct compute-tier
compositions mirroring their infallible twins, instead of routing
through the FASC pipeline (LazyExpr tree + TLS evaluator + domain
dispatch) per call. Same engines, same formulas, one downscale; results
are bit-identical to the infallible methods on in-domain inputs, and the
0.4.27 error contract is unchanged (`DomainError` for |x|>1 asin/acos,
x<1 acosh, |x|≥1 atanh; `TierOverflow` on storage overflow; `asin(±1)`
= ±π/2 exactly; tanh saturates to exactly 1 at the exp ceiling). Gated
by `tests/try_direct_bypass_validation.rs`.

Two silent-wrong-value defects flushed out by the new gate (both in the
0.5.0 wrap class):

- The imperative π/2 constant on the realtime profile cast a Q64.64
  quantity straight to i64: π/2·2^64 wrapped negative, silently
  corrupting every imperative `acos` on that profile (the FASC path was
  unaffected, which is why nothing gated it). Now a rounded shift to the
  profile's compute scale.
- `sinh`/`cosh` at the exp overflow sentinel: the q128_128 exp engine's
  sentinel (`i128::MAX` at storage scale) equals the storage maximum, so
  it downscaled CLEANLY into a plausible-wrong result: `cosh(180)` on
  balanced returned ~storage-max/2 as `Ok`, and the FASC pipeline's own
  ceiling guard (`== compute max`) had the same blind spot on that
  profile. A shared per-profile sentinel predicate now guards both
  paths: infallible sinh/cosh panic, `try_` variants return
  `TierOverflow`, FASC cosh/tanh use the corrected check.

### Fixed: UGOD ladder top (0.5.0 item 1): exact or loud, never wrapped

Verdict of the promotion audit: mid-ladder promotion (binary tiers 1→4)
was always correct, but the TOP of every ladder wrapped silently.
Contract now gated by `tests/ugod_promotion_validation.rs` on every
profile (and in CI): arithmetic on representable inputs either returns
the EXACT value (via a wider tier or the symbolic domain) or fails
loud.

- Binary Tier-4/5 multiply truncated its wide product unchecked
  (`1e20 × 1e20` on balanced returned 1.318e38: the product mod 2^256);
  Tier-4/5/6 add/sub used bare wrapping operators (`9e18 + 9e18` on
  embedded returned 0.0). All top tiers now use checked arithmetic with
  4→5→6 promotion arms.
- Storage narrowing of promoted results (`binary_to_storage`) used bare
  casts; now fits-checked, and the FASC binary arms fall back to the
  exact rational path on `TierOverflow`: the true ladder top.
- Divide branched into per-tier code before distinguishing zero divisors
  from quotient overflow, mislabeling overflow as `DivisionByZero`; the
  zero check now happens once at ladder entry, and overflow falls back
  to the exact rational quotient.
- The symbolic ladder's multiply "promotion" retried at the same i128
  width (Huge×Huge could never reach the existing Massive/I256 tier);
  `divide_mixed_tiers` returned the target-tier attempt without
  escalating (a quotient can need a wider tier than either operand:
  9e18 ÷ 1e-9 = 9e27 needs Huge). Both now climb the ladder.
- FASC's symbolic/ternary → binary coercion (`to_binary_storage`)
  shifted i128 numerators before any range check: a symbolic 1e20
  coerced on embedded wrapped mod 2^64 into a PLAUSIBLE WRONG value.
  Replaced with a checked nearest-ties-+∞ conversion at tier N+1 width.
- Fractional literals beyond a narrow profile's decimal cap silently
  truncated at parse: `"0.000000001"` on realtime parsed to exactly 0,
  turning later divisions into division-by-zero. Such literals now fall
  back to the exact Symbolic domain (the fractional twin of the item-0
  integer fallback below).
- The scientific (Q256.256) formatter squeezed integer parts through
  i128, so any result ≥ 2^127 DISPLAYED as its value mod 2^128 even when
  the stored bits were exact. Integer parts now print digit-at-a-time at
  I256 width.
- A sqrt compute-tier multiply helper assumed non-negative operands, but
  Newton's 3 − S·y² factor goes negative on seed overshoot (caught by
  the new 0b debug_asserts on scientific); the helper is now sign-safe.

### Fixed

- **Realtime FASC decimal results lost half their digits at
  materialization** ("the cosine plateau that wasn't"): DecimalCompute
  values were materialized at a fixed `DECIMAL_STORAGE_MAX_DP − 2` in
  Display/to_decimal_string/to_rational: 2 harmless digits of slack on
  wide profiles, but dp 4 → 2 on realtime, so cos(0.1) = 0.9952 rendered
  as "1.00" and masqueraded as a sin/cos kernel plateau (the kernels were
  bit-perfect all along). Now adaptive: full MAX_DP first, stepping down
  only when the magnitude needs fewer decimals (checked, deterministic).
  Wide profiles regain their two withheld display digits; realtime passes
  its original strict tolerances again.
- UGOD binary tier divide mis-signed its rounding bump for exact
  quotients in (−1, 0) raw units (branched on `quotient < 0`, which is 0
  there): e.g. an exact −0.75-ulp quotient rounded to **+1** raw instead
  of −1. Fixed by deriving the sign from the operands; covered by the
  unification gate's sub-ulp regression case.
- Ternary Tier-4 negation used `saturating_neg` (the one tier silently
  absorbing the binary-MIN edge); now fail-loud like every other tier.

- Integer literals beyond the profile's binary integer range now fall
  back to the Symbolic domain instead of failing parse with `Overflow`
  (UGOD's ladder top never fails). Pre-fix, realtime could not parse
  `32768`+ at Q16.16, so `1000000 * 0.001` errored there while
  succeeding on every other profile. Regression-tested on every profile;
  the router/domain integration suites now also run on realtime+compact
  in CI, which is how this stayed hidden.

## [0.4.34] - 2026-08-14

### Added

- **Ternary routing column**: the fractal router's classifier has computed
  `TERNARY_BIT` since v0.4.0 with no table column consuming it; ternary-exact
  values (denominator 3^k: 1/3, 2/3, 100/3…) fell through to the symbolic
  rational fallback. `DomainChoice::Ternary` now exists and cross-domain
  **Add/Sub** of 3-adic operands routes into the balanced-ternary domain,
  where it is exact by construction (sums of 3-adic values stay 3-adic).
  **Mul/Div are deliberately excluded**: products multiply denominators past
  the tier scale, where ternary truncates while symbolic stays exact, and
  the 4-bit class mask cannot see exponents: routing them would let routing
  change results. Full reasoning: `docs/design/TERNARY_ROUTING_COLUMN.md`.
- Coercion failure falls back silently to the previous route: on narrow
  profiles a large 3-adic value can overflow ternary storage, and the
  router must never introduce a failure the old path did not have.
- The classifier now reads **Symbolic operands' denominators directly**
  (they carry no shadow: the rational itself is richer than any shadow).
  Side effect beyond ternary: symbolic operands with 2-adic/10-adic
  denominators can now coerce into Binary/Decimal on Add/Sub/Mul/Div where
  both operands are exact there: exactness preserved in every such case.
- Measured (embedded, release, full evaluate pipeline including literal
  parsing): routed `0t2 + 1/3` 334 ns vs rational-fallback `0t2 + 1/7`
  355 ns (~1.06×). The win at expression scale is modest: parsing
  dominates; the value is architectural: the 3-adic exactness class now
  reaches its domain and results stay in fixed-point form.

### Fixed

- `convert_to_ternary` (output-mode/coercion conversion) stored tier-3
  raws through an **unchecked** narrowing cast in its tier-3 and fallback
  arms: on realtime/compact, converting e.g. `1/3` with output mode
  `ternary` silently wrapped (same defect class as the 0.4.33
  `ternary_to_storage` fix). Both arms now use the checked conversion:
  loud `TierOverflow`, never a wrap. Pinned by test.

- `multiply_ternary_tq256_256` (Tier-6 balanced ternary) passed operands
  straight into the unsigned `mul_to_i2048`, so any negative operand
  produced a product with corrupted sign extension. Now sign-wrapped
  (magnitudes in, sign restored) per the widening-multiply convention.
- `I2048::Mul`'s I512 fast path had the same defect via `mul_to_i1024`,
  corrupting Tier-6 ternary division of negative values. Sign-wrapped the
  same way. Scientific-profile transcendental validation (18/18) re-run
  clean after the change.

### Documentation

- CONTRACT.md's rounding table was wrong on two rows and is now stated
  per-path, verified against source: binary multiply is round-half-even
  in the imperative `FixedPoint` kernel but ties-toward-+∞ in the
  canonical/UGOD tier path: **1 ULP apart on exact half-ULP ties**
  (measured; the one known exception to path independence, scheduled for
  unification in 0.5.0); `DecimalFixed<D>` is banker's (not half-away),
  and the canonical decimal domain's multiply is exactness-preserving.
  README and routing guide updated to carry the same caveat.

### Testing

- Wide-tier ternary coverage closing the 0.4.33 gaps: Tiers 2–6
  arithmetic against exact integer models with every sign combination
  (the adversarial axis for the unsigned-word defect class), wide
  `ternary_to_storage` arm unit tests (exact-or-loud on every profile),
  FASC-transcendental-on-ternary path equivalence, and the pin that `0t`
  literals cap at Tier 3. All five profiles green.

## [0.4.33] - 2026-08-11

### Added

- **Balanced ternary contract + dedicated validation** (closes the
  long-standing ternary-coverage gap): `docs/design/BALANCED_TERNARY_CONTRACT.md`
  specifies both shipping representations (native trits for packing and
  zero-multiply inference kernels; radix-3 scaled integers for the UGOD
  tiers, with TQ1.9 as the window-enforced hybrid), verified operation
  semantics, and the tie-free rounding theorem (3^m is odd, so
  round-to-nearest onto a 3-adic grid never ties; trit truncation IS
  round-to-nearest). Suites: `ternary_domain_validation` (trit-vector
  reference oracle, exhaustive small-range equivalence, boundary families,
  theorem tests) and `ternary_path_equivalence` (UGOD promotion at
  raw-overflow boundaries, canonical-vs-imperative equivalence over `0t`
  literals, cross-domain coercion neutrality, conversion pins). New
  `ternary-domain` CI workflow runs both on every push.
- `FixedPoint::inv_sqrt` / `try_inv_sqrt`: 1/√x at the compute tier
  (square root and reciprocal both at tier N+1, one rounding at the final
  downscale). `try_` returns `DomainError` for x ≤ 0, `TierOverflow` if
  the result exceeds storage. Reciprocal norms are the target use:
  one `inv_sqrt` plus N multiplies replaces N per-component divisions in
  normalization (division is ~200× a multiply at Q64.64).
- `fused::inv_sqrt_sum_sq`: 1/√(Σ vᵢ²) entirely at the compute tier; the
  reciprocal-norm form vector normalization actually wants. Panics on a
  zero vector (documented), matching the fused family's conventions.
  Both additions are purely additive: no existing output bits move.

### Fixed

- `UniversalTernaryFixed::from_str` dropped the sign of `-0.x` inputs
  (`"-0".parse::<i64>()` is 0): `-0.5` parsed as +0.5. The sign is now
  stripped once and the parsed magnitude negated: identical results for
  every input with a nonzero integer part, correct results for `-0.x`.
- `ternary_to_storage` narrowed tier raws with bare `as` casts: on the
  realtime/compact profiles a Tier-2+ ternary value silently wrapped
  (`0t3281` displayed as `-11.56021` on realtime). Conversion is now
  checked end-to-end and returns `TierOverflow` instead: wrap-defect
  class, same family as the 0.4.28–0.4.31 fixes. Note the asymmetry pinned
  by test: values reached by arithmetic stay at their operand tier
  (0t3280 + 0t1 is fine everywhere), while `from_str` window-gates
  literals upward (a bare `0t3281` literal errors loudly on realtime).

## [0.4.32] - 2026-08-01

### Added

- `g_math::compute_tier` (feature `inference`): public compute-tier
  (tier N+1) transcendentals over raw `ComputeStorage` values at
  2·FRAC_BITS fractional precision: `exp`, `ln`, `sqrt`, `sinhcosh`
  primitives plus `sigmoid`, `softplus`, `ln1p` compositions, with
  `from_fixed`/`to_fixed`/`try_to_fixed` conversions and the `one`/
  `ceiling` constants. These are the same engines every other API path
  uses (results are path-independent with the canonical and imperative
  surfaces (pinned by test)) exposed so wide-precision inference
  consumers no longer re-derive integer-only exp/sigmoid/softplus/ln1p
  on top of the storage-tier API. The format matches the wide-output
  `matvec_q2f` family (0.4.31), so those accumulators feed these
  functions directly with no conversion.
- Contract: `exp` saturates at `ceiling()` and never wraps; `ln`/`sqrt`/
  `ln1p` panic on domain violations; `to_fixed` panics (and
  `try_to_fixed` returns `None`) when a value does not fit storage;
  nothing wraps silently. `sigmoid` and `softplus` use sign-split stable
  forms whose intermediates cannot overflow for any input.
- Validation (`tests/compute_tier_validation.rs`): mpmath 60-digit
  references at q16_16 and q32_32, gated at measured maxima: storage
  level exact (0 LSB) for every function at both profiles; compute-tier
  raw kernel output within 0–4 ULP (primitives) / 1–5 ULP (compositions)
  of the true value. Plus path-independence, saturation, domain-panic,
  and symmetry-identity gates.
- Free `ln_at_compute_tier` kernel wrapper (crate-internal), completing
  the exp/sqrt/sinhcosh free-function set.

## [0.4.31] - 2026-07-22

### Added

- Wide-output `matvec_q2f` family for all four TQ1.9 forms (`TQ19Matrix`,
  `RowScaledTQ19`, `HybridTQ19`, `PlanarTQ19`), plus `tq19_dot_q2f`, gated
  to the q16_16/q32_32 profiles under `inference`. Each returns the exact
  row accumulator at 2·FRAC_BITS fractional precision with exactly one
  rounding (truncation toward zero of `acc·2^FRAC_BITS / SCALE`) instead
  of rounding to storage in the epilogue: for consumers whose signal sits
  below the storage rounding floor (e.g. fine-grained-MoE expert outputs).
  Inner loops, SIMD dispatch, and rayon parallelism are unchanged; zero
  cost on the narrow path.
- Narrowing contract, pinned by property tests on every gated profile:
  `q2f / (1 << FRAC_BITS)` (Rust truncating division) reproduces the
  narrow `matvec` bit-for-bit for `TQ19Matrix`/`HybridTQ19`/`PlanarTQ19`
  (nested truncation toward zero is exact). `RowScaledTQ19::matvec_q2f`
  applies the per-row scale to the wide dot: strictly more precise, so
  narrowing it may differ from the narrow path by ±1 storage LSB for
  non-unit scales (exact for unit scales); its wide value is pinned
  against an independent i128 oracle. Out-of-range wide outputs fail
  loud, never wrap.

## [0.4.30] - 2026-07-22

### Added

- `tq19::RowScaledTQ19` ("TQ1.9-R"): TQ1.9 with one quantization scale per
  row (Maniference O27 contribution). Per-row scales adapt the quantization
  step to each row's own max: measured on Mixtral-8x7B, matvec output error
  drops ~20× and wrong-expert routing drops 6.4% → 0.22% of tokens, at
  unchanged 2 bytes/weight plus one i128 multiply-shift per output element.
  Matvec reuses the existing SIMD `tq19_dot` verbatim. Gated to the q16_16
  and q32_32 profiles (wider profiles would need bigint scale arithmetic).
  Review hardening on merge: the scaled output now fails loud if it exceeds
  the storage range instead of wrapping, and an independent i128-oracle test
  (no shared code with the SIMD path) pins matvec and scale application.

## [0.4.29] - 2026-07-22

Root-cause fix for a latent exp-overflow corruption reported by the
Maniference project (their O26: Mixtral-8x7B expert gates reach ±70 and
tripped a path dense models never reach).

### Fixed

- `downscale_q64_to_q32` (the Q64.64 → compute-tier downscale used by the
  q16_16 and q32_32 exp/trig wrappers) wrapped oversized results via a plain
  `as i64` cast, including the exp overflow sentinel `i128::MAX`. It now
  saturates to `i64::MAX`/`i64::MIN`, so oversized results stay detectable
  and every later storage downscale reports them instead of materializing
  wrapped garbage. Measured pre-fix corruption at Q22.10: `fused::silu(-70)`
  returned `-70` (the gate value passed through unsquashed: a 200× residual
  spike in Maniference's Mixtral run) and `tanh(25)` returned `-1`.
- Ceiling guards for every exp consumer whose follow-up add could wrap (or
  panic in debug builds) on a saturated/sentinel exp:
  `fused::silu` returns 0 (correctly rounded for every such input),
  `FixedPoint::tanh` returns 1 (likewise), `FixedPoint::cosh` and the FASC
  cosh path fail loud (`cosh overflow` / `TierOverflow`), the FASC tanh path
  returns 1, and `sinhcosh_at_compute_tier` pins the cosh sum at the ceiling
  so materialization reports overflow. Softmax paths were audited and are
  safe (max-subtraction bounds exp inputs at 1).
- `tests/tq19_bench.rs` weight generator overflowed i32 at 4096×4096 in
  debug builds (pre-existing, test-infrastructure only).

### Added

- Regression tests: exp monotonicity property over the full storage range
  (the wrap broke monotonicity), silu deep-negative (−30 … −100, all
  profiles), tanh saturation/sentinel region = 1 (all profiles), and FASC
  try_tanh/try_cosh behavior on narrow profiles.

## [0.4.28] - 2026-07-18

### Fixed

- `round_to_storage` (shared downscale for the fused kernels, linalg dot,
  and decompositions): results exceeding the storage tier's range now
  panic (matching the infallible imperative transcendentals) instead of
  silently wrapping via the old shift-and-cast fallback. On narrow
  profiles the wrap produced garbage (e.g. a squared distance of 520,000
  returned as −4288 on Q16.16). Valid-range results are unaffected.
- Fused test suite made multi-profile compliant: test magnitudes fit the
  narrowest profile, and tolerances are representable at Q22.10 (the old
  `0.0001` tolerance rounded to 0 raw, failing identical values) and
  account for input quantization of non-representable decimal literals.

## [0.4.27] - 2026-07-11

### Fixed

- `FixedPoint::try_ln` / `FixedPoint::try_sqrt`: out-of-domain inputs
  (ln(x ≤ 0), sqrt(x < 0)) again return `OverflowDetected::DomainError` as
  documented. Since the v0.4.0 direct-engine-call rewrite these methods
  bypassed the FASC domain checks and misreported out-of-domain input as
  `TierOverflow` (the raw engine's MIN sentinel failing the storage
  downscale). The domain check now runs before the engine call on the
  direct path; valid inputs are unaffected.

## [0.4.26] - 2026-07-11 (unpublished)

The U1 consumer asks from gHyper/gFile (see their ROADMAPs): the fused
no-transcendental kernels that hyperbolic metric trees and Möbius-ratio
distance kernels score with.

### Added

- `fused::euclidean_distance_squared`: Σ (a−b)² at compute tier, no sqrt.
  The no-transcendental half of `euclidean_distance`: squared-space VP-tree
  scoring and Möbius-ratio numerators need only the squared value, and a
  fixed-point sqrt (~15 µs at Q64.64) immediately re-squared is the
  dominant waste in those kernels.
- `fused::dot`: Σ a·b at compute tier; replaces consumers' storage-tier
  hand-rolled accumulators (wrap-prone for large coordinates/dimensions).
- `fused::mobius_denominator_sq`: |1 − p̄q|² = 1 − 2⟨p,q⟩ + |p|²·|q|²
  fused end-to-end (one downscale). With `euclidean_distance_squared` this
  gives consumers the one-sqrt Poincaré kernel: r = √(dist²/den²).

## [0.4.25] - 2026-07-09

Hardening of the trit-plane inference formats and the fused attention op, and a
documentation overhaul to the Geodineum README standard.

> Note: the changelog was not maintained between 0.1.0 and 0.4.24; see the git
> history and `ROADMAP.md` for the intervening milestones (five profiles, TQ1.9,
> decimal transcendentals, fractal router, geometric extension).

### Fixed

- `fused::softmax_mix`: the `Σⱼ eⱼ·vⱼ` numerator and the exp-sum now accumulate
  with overflow detection and return `OverflowDetected::TierOverflow` instead of
  silently wrapping on long-context × large-activation inputs.
- `fused::softmax_mix`: value-row length mismatch is now a hard `assert!` (was a
  `debug_assert!`), so a ragged value matrix cannot silently mix wrong dimensions
  in release builds.

### Added

- `I1024::checked_add`: signed overflow-detecting addition (mirrors
  `I256`/`I512`), enabling overflow-safe compute-tier accumulation on the
  scientific profile.
- `softmax_mix` oracle tests (`tests/fused_ops_validation.rs`): exact-rational
  uniform-mean (long-n, the storage-floor survival property) plus mpmath 60-digit
  references for distinct-scores and near-one-hot mixes, validated on all five
  profiles.
- CI workflow `fused-tq19-precision.yml`: fused oracle and PlanarTQ19/HybridTQ19
  bit-exactness across all five profiles, plus the realtime Q22.10 floor branch.
- Documentation: `README.md` rewritten to the Geodineum README standard; per-layer
  guides under `docs/`; `CONTRACT.md` (integration/precision/determinism contract)
  and `CONTRACT.scn.md` (agent primer); generated `PUBLIC_API.md` with its
  regenerable extractor `scripts/gen-public-api.rs`.

### Changed

- `HybridTQ19` exhaustive split test tightened to the true invariant `hi ∈ [-13, 13]`.

## [0.1.0] - 2026-03-01

Initial open-source release.

### Core

- **FASC** (Fixed-Allocation Stack Computation) pipeline: `LazyExpr` tree builder with operator overloading, thread-local `StackEvaluator` with fixed-size workspace (4KB-64KB)
- **UGOD** (Universal Graceful Overflow Delegation): automatic 6-tier promotion across all domains, with symbolic rational as guaranteed-success fallback
- **Tier N+1** precision strategy: all transcendentals compute one tier above storage, single downscale at materialization
- **BinaryCompute chain persistence**: chained transcendentals stay at compute tier throughout, preventing cumulative precision loss
- **CompactShadow** precision preservation: 0-32 byte exact rational shadow on all non-symbolic values, propagated through arithmetic

### Domains

- **Binary fixed-point**: Q64.64 / Q128.128 / Q256.256 with 18 transcendental functions via tier N+1 computation
- **Decimal fixed-point**: exact base-10 arithmetic (0.1 + 0.2 = 0.3), 6-tier UGOD
- **Symbolic rational**: exact a/b arithmetic with 7-tier storage hierarchy (i8 to I512)
- **Balanced ternary**: base-3 fixed-point with 6-tier UGOD

### Transcendental Functions (18 total)

- **Dedicated algorithms**: exp, ln, sqrt, sin/cos, atan; each with tier N+1 table-driven implementations
- **FASC-composed**: tan, pow, asin, acos, atan2, sinh, cosh, tanh, asinh, acosh, atanh
- **AVX2 SIMD**: Q64.64 multiply hotpath with scalar fallback

### Mode Routing

- 25 compute:output combinations via `set_gmath_mode("binary:decimal")`
- Thread-local `Cell<GmathMode>` for zero-contention mode switching

### Profiles

- `GMATH_PROFILE=embedded`: Q64.64, 19 decimals, scalar
- `GMATH_PROFILE=performance`: Q64.64, 19 decimals, AVX2-optimized
- `GMATH_PROFILE=balanced`: Q128.128, 38 decimals
- `GMATH_PROFILE=scientific`: Q256.256, 77 decimals

### Build System

- Pure-Rust `build.rs` with zero external runtime dependencies
- Algorithmic constant generation: Machin's formula (pi), factorial series (e), continued fractions (sqrt2)
- 3-stage x 1024 entry lookup tables per tier for exp, ln, and trig
- Build cache: skip regeneration when source/profile unchanged

### Validation

- 60,860 arithmetic reference points (mpmath-verified, 4 domains x 4 operations)
- 16,974 transcendental reference points (18 functions x 1,000+ values)
- 288 mode routing test points (12 modes x 24 cases)
- 0 lossy results across all mode combinations

### Cross-Platform

- Bit-identical results across all architectures (x86, ARM, RISC-V)
- Zero floating-point contamination (f32/f64 forbidden in internal logic)
- Consensus-safe for blockchain, financial auditing, scientific reproducibility
