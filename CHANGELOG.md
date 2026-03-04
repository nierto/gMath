# Changelog

All notable changes to gMath will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-03-01

Initial open-source release.

### Core

- **ZASC** (Zero-Allocation Stack Computation) pipeline: `LazyExpr` tree builder with operator overloading, thread-local `StackEvaluator` with fixed-size workspace (4KB-64KB)
- **UGOD** (Universal Graceful Overflow Delegation): automatic 6-tier promotion across all domains, with symbolic rational as guaranteed-success fallback
- **Tier N+1** precision strategy: all transcendentals compute one tier above storage, single downscale at materialization
- **BinaryCompute chain persistence**: chained transcendentals stay at compute tier throughout, preventing cumulative precision loss
- **CompactShadow** precision preservation: 0-32 byte exact rational shadow on all non-symbolic values, propagated through arithmetic

### Domains

- **Binary fixed-point**: Q64.64 / Q128.128 / Q256.256 with 0 ULP across all 18 transcendental functions
- **Decimal fixed-point**: exact base-10 arithmetic (0.1 + 0.2 = 0.3), 6-tier UGOD
- **Symbolic rational**: infinite-precision a/b arithmetic with 7-tier storage hierarchy (i8 to I512)
- **Balanced ternary**: base-3 fixed-point with 6-tier UGOD

### Transcendental Functions (18 total, 0 ULP)

- **Dedicated algorithms**: exp, ln, sqrt, sin/cos, atan — each with tier N+1 table-driven implementations
- **ZASC-composed**: tan, pow, asin, acos, atan2, sinh, cosh, tanh, asinh, acosh, atanh
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
