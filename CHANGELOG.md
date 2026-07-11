# Changelog

All notable changes to gMath will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
