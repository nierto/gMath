# gMath Roadmap

Current version: **0.3.90**

This document tracks planned work and known gaps. Items are grouped by priority, not by timeline. Nothing here is a promise — this is a working list for a solo-maintained project.

---

## Delivered

### v0.3.0 — Five profiles, 0 ULP

| Profile | Storage | Compute | Bytes | Digits | Status |
|---------|---------|---------|-------|--------|--------|
| realtime | Q16.16 (i32) | Q32.32 (i64) | 4 | 4 | **0 ULP** |
| compact | Q32.32 (i64) | Q64.64 (i128) | 8 | 9 | **0 ULP** |
| embedded | Q64.64 (i128) | Q128.128 (I256) | 16 | 19 | **0 ULP** |
| balanced | Q128.128 (I256) | Q256.256 (I512) | 32 | 38 | **0 ULP** |
| scientific | Q256.256 (I512) | Q512.512 (I1024) | 64 | 77 | **0 ULP** |

All profiles use true tier N+1 computation. 18 transcendentals, 4 domains (binary, decimal, ternary, symbolic), FASC zero-allocation stack computation, UGOD tiered overflow delegation, BinaryCompute chain persistence.

### v0.3.89 — TQ1.9 standalone module

Dedicated inference module with AVX2 SIMD, rayon row-parallel dispatch, batch matvec. `TQ19Matrix`, `tq19_dot`, `trit_dot`, packed trit operations, `TRIT_DECODE_TABLE`. 38 tests.

### v0.4.0 — Matrix chain persistence, fused ops

- `LazyMatrixExpr` — 14-variant enum, operator overloading, recursive evaluator at ComputeMatrix tier
- `DomainMatrix` — StackValue-tagged matrix, 4 domains + cross-domain routing
- Fused transcendental paths — `evaluate_sincos()`, identity short-circuits (`exp(ln(x))` -> x)
- Fused compute-tier ops — `sqrt_sum_sq`, `euclidean_distance`, `softmax`, `rms_norm_factor`, `silu`

### v0.3.90 — Configurable FRAC_BITS, native transcendental dispatch, inference feature gate

- `GMATH_FRAC_BITS` env var for realtime profile (e.g., Q8.24 via `GMATH_FRAC_BITS=24`)
- Native transcendental dispatch: Q16.16/Q32.32 use hardware i128 instead of software I256
- Fixed `ln_q64_64_native` algorithm bug (additive -> multiplicative decomposition)
- `FixedPoint::sincos_wide(i64)` for wide-range RoPE position encoding
- TQ1.9 gated behind `inference` feature (replaces `parallel`)
- Decimal-to-binary rounding fix (round-to-nearest instead of truncation)

---

## Next: 0.5.0 — Decimal transcendentals + correctness audit

### 1. UGOD multi-tier promotion verification

**Priority:** HIGH — correctness concern
**Effort:** ~200 lines, 1 session

Verify that UGOD overflow promotion tries at least 2 subsequent tiers before falling back to symbolic rational. Current behavior may promote to symbolic after a single tier overflow, bypassing intermediate tiers that would succeed. Either confirm the code is correct (document the guarantee) or fix it.

### 2. Decimal domain transcendentals

**Priority:** HIGH — unlocks an entire application domain
**Effort:** ~3,000 lines, 3-4 sessions

Native decimal exp/ln/sqrt/sin/cos/tan/atan at tier N+1 with decimal-specific tables. This is the highest-impact architectural addition remaining:

**What it enables:**
- **DecimalCompute chain persistence** — the decimal equivalent of BinaryCompute. Values like 0.1, 0.01, 1/3 that are exact in decimal but lossy in binary stay exact through transcendental chains. Eliminates the representation error class we encountered with Q8.24 ln(0.1).
- **Financial-grade computation** — compound interest, Black-Scholes, amortization schedules, tax calculations with exact decimal arithmetic end-to-end. No binary round-trip.
- **Multi-domain composability** — all 13 composed transcendentals (sinh, cosh, tanh, asin, acos, pow, etc.) become natively multi-domain. FASC chain persistence works per-domain.
- **Architectural precedent** — establishes the pattern for symbolic rational transcendentals (future: ultra-precision mode where all intermediates are exact rationals).

**Requires:** decimal-specific table generation in `build.rs`, decimal tier N+1 compute infrastructure, DecimalCompute variant in StackValue.

### 3. Stack evaluator `profile_dispatch!` macro

**Priority:** MEDIUM — reduces maintenance burden, prevents cfg copy-paste bugs
**Effort:** ~750 lines, 2 sessions

Extract macro to replace the 5-way `#[cfg(table_format)]` blocks across all stack evaluator submodules. Each profile-conditional function currently has 5 copy-paste arms. A macro reduces this to 1 invocation per function and makes future profile additions mechanical.

---

## Future — High Priority

### Batch/vectorized API

SIMD-friendly array processing for FixedPoint operations beyond TQ1.9. Bulk exp, sqrt, and arithmetic over vectors would accelerate softmax, RMSNorm, and embedding decode. Compelling with Q32.32 (8x i32 per AVX2 register) and Q16.16/Q8.24 (16x i16).

### Ternary test coverage

Balanced ternary arithmetic lacks a dedicated validation suite. The domain works but needs stress-testing against reference values. Low effort, fills a known quality gap.

### Imperative geometry methods — UGOD + FASC integration

Upstream `square()`, `reciprocal()`, `powi()`, `manhattan_distance()`, `mul_vector()` etc. as first-class UGOD-dispatched, FASC-computed methods. ~800 lines.

### Symbolic rational transcendentals

The third compute domain after Binary and Decimal. Transcendental chains where ALL intermediates are exact rational numbers (num/den BigInt pairs). Unbounded precision — no ULP concept, just exact arithmetic until final materialization. Ultra-precision mode for scientific computing and formal verification.

### Tensor decompositions (L2B)

Truncated SVD, HOSVD/Tucker, CP/ALS for model compression. Optimization-tier features for distributed inference.

### Fractal topology FASC integration

Wire the fractal topology engine into FASC `parse_literal()` for geometric domain routing. Replace syntactic routing with data-driven routing. Not urgent — current routing works correctly.

### n-D Clifford algebra (L4B)

Vahlen matrices over Cl(n,0,1). High novelty — no zero-float Clifford algebra in Rust.

### Custom FRAC_BITS for non-realtime profiles

Extending `GMATH_FRAC_BITS` to compact (i64), embedded (i128), and higher profiles. Requires verifying that `COMPUTE_FRAC_BITS = 2 * FRAC_BITS` does not exceed the native transcendental tier's capacity. Deferred until demand.

### Public API stabilization

Pre-1.0 audit of exports, feature gating, StackValue extraction methods.

---

## Non-goals

- **Floating-point interop beyond convenience**: `to_f64()`/`from_f64()` exist for user convenience. Internal float usage is architecturally forbidden.
- **Dynamic precision selection**: Profiles are compile-time. Runtime tier selection within a profile is handled by UGOD, but the base storage tier is fixed at build time.
- **GPU compute**: The library targets CPU determinism. GPU offload would compromise the cross-platform bit-identical guarantee.
