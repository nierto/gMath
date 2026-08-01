# Changelog

All notable changes to gMath will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.32] - 2026-08-01

### Added

- `g_math::compute_tier` (feature `inference`) — public compute-tier
  (tier N+1) transcendentals over raw `ComputeStorage` values at
  2·FRAC_BITS fractional precision: `exp`, `ln`, `sqrt`, `sinhcosh`
  primitives plus `sigmoid`, `softplus`, `ln1p` compositions, with
  `from_fixed`/`to_fixed`/`try_to_fixed` conversions and the `one`/
  `ceiling` constants. These are the same engines every other API path
  uses — results are path-independent with the canonical and imperative
  surfaces (pinned by test) — exposed so wide-precision inference
  consumers no longer re-derive integer-only exp/sigmoid/softplus/ln1p
  on top of the storage-tier API. The format matches the wide-output
  `matvec_q2f` family (0.4.31), so those accumulators feed these
  functions directly with no conversion.
- Contract: `exp` saturates at `ceiling()` and never wraps; `ln`/`sqrt`/
  `ln1p` panic on domain violations; `to_fixed` panics (and
  `try_to_fixed` returns `None`) when a value does not fit storage —
  nothing wraps silently. `sigmoid` and `softplus` use sign-split stable
  forms whose intermediates cannot overflow for any input.
- Validation (`tests/compute_tier_validation.rs`): mpmath 60-digit
  references at q16_16 and q32_32, gated at measured maxima — storage
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
  of rounding to storage in the epilogue — for consumers whose signal sits
  below the storage rounding floor (e.g. fine-grained-MoE expert outputs).
  Inner loops, SIMD dispatch, and rayon parallelism are unchanged; zero
  cost on the narrow path.
- Narrowing contract, pinned by property tests on every gated profile:
  `q2f / (1 << FRAC_BITS)` (Rust truncating division) reproduces the
  narrow `matvec` bit-for-bit for `TQ19Matrix`/`HybridTQ19`/`PlanarTQ19`
  (nested truncation toward zero is exact). `RowScaledTQ19::matvec_q2f`
  applies the per-row scale to the wide dot — strictly more precise, so
  narrowing it may differ from the narrow path by ±1 storage LSB for
  non-unit scales (exact for unit scales); its wide value is pinned
  against an independent i128 oracle. Out-of-range wide outputs fail
  loud, never wrap.

## [0.4.30] - 2026-07-22

### Added

- `tq19::RowScaledTQ19` ("TQ1.9-R") — TQ1.9 with one quantization scale per
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
  `as i64` cast — including the exp overflow sentinel `i128::MAX`. It now
  saturates to `i64::MAX`/`i64::MIN`, so oversized results stay detectable
  and every later storage downscale reports them instead of materializing
  wrapped garbage. Measured pre-fix corruption at Q22.10: `fused::silu(-70)`
  returned `-70` (the gate value passed through unsquashed — a 200× residual
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
  panic — matching the infallible imperative transcendentals — instead of
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

- `fused::euclidean_distance_squared` — Σ (a−b)² at compute tier, no sqrt.
  The no-transcendental half of `euclidean_distance`: squared-space VP-tree
  scoring and Möbius-ratio numerators need only the squared value, and a
  fixed-point sqrt (~15 µs at Q64.64) immediately re-squared is the
  dominant waste in those kernels.
- `fused::dot` — Σ a·b at compute tier; replaces consumers' storage-tier
  hand-rolled accumulators (wrap-prone for large coordinates/dimensions).
- `fused::mobius_denominator_sq` — |1 − p̄q|² = 1 − 2⟨p,q⟩ + |p|²·|q|²
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

- `I1024::checked_add` — signed overflow-detecting addition (mirrors
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

- **Dedicated algorithms**: exp, ln, sqrt, sin/cos, atan — each with tier N+1 table-driven implementations
- **FASC-composed**: tan, pow, asin, acos, atan2, sinh, cosh, tanh, asinh, acosh, atanh
- **AVX2 SIMD**: Q64.64 multiply hotpath with scalar fallback

### Mode Routing

- 25 compute:output combinations via `set_gmath_mode("binary:decimal")`
- Thread-local `Cell<GmathMode>` for zero-contention mode switching

### Profiles

- `GMATH_PROFILE=embedded` — Q64.64, 19 decimals, scalar
- `GMATH_PROFILE=performance` — Q64.64, 19 decimals, AVX2-optimized
- `GMATH_PROFILE=balanced` — Q128.128, 38 decimals
- `GMATH_PROFILE=scientific` — Q256.256, 77 decimals

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
