# gMath Roadmap

Current version: **0.4.0**

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

### v0.3.90 — Configurable FRAC_BITS, native transcendental dispatch, inference feature gate

- `GMATH_FRAC_BITS` env var for realtime profile (e.g., Q8.24 via `GMATH_FRAC_BITS=24`)
- Native transcendental dispatch: Q16.16/Q32.32 use hardware i128 instead of software I256
- Fixed `ln_q64_64_native` algorithm bug (additive -> multiplicative decomposition)
- `FixedPoint::sincos_wide(i64)` for wide-range RoPE position encoding
- TQ1.9 gated behind `inference` feature (replaces `parallel`)
- Decimal-to-binary rounding fix (round-to-nearest instead of truncation)

### v0.4.0 — Decimal transcendentals, fractal router, direct engine calls, gmath! macro

**Decimal transcendentals (native, 0 ULP):**
- 5 native engines: exp (4-stage table decomposition), ln (atanh), sqrt (Newton-Raphson), sin/cos (Cody-Waite + Machin pi), atan (half-angle)
- DecimalCompute StackValue variant + full FASC dispatch
- DecimalFixed imperative type: 18 transcendental methods wired to native engines (eliminated binary round-trip)
- 14 ULP validation tests + 24 FASC integration tests + ~9500 mpmath reference points

**Fractal topology router:**
- Shadow-based operand classifier (~15 ns) — factors CompactShadow denominator to determine domain exactness
- Compile-time routing table (5.25 KB .rodata, 21 ops x 16 x 16 classes) — O(1) lookup
- Cross-domain coercion in arithmetic — `gmath("0.1") + gmath("255")` routes to Decimal (was Symbolic fallback)
- Tree walker: `route_expression(&LazyExpr) -> OperandClass`, O(N) bottom-up

**Decimal 4-stage exp tables:**
- `exp(x) = exp(k) * exp(d1/10) * exp(d2/100) * exp(d3/1000) * exp(r)` — 71 cached entries
- 4.1x speedup on decimal exp (43K -> 177K ops/s), binary gap narrowed 8.6x -> 2.1x

**Direct binary engine calls (FASC bypass):**
- FixedPoint: exp/ln/sqrt/sin/cos/atan use `direct_unary` pattern — upscale -> engine -> downscale
- ~65 ns saved per call vs FASC pipeline (no LazyExpr, no TLS, no StackValue boxing)

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

**Composed transcendental direct bypass (complete FASC elimination):**
- ALL 18 FixedPoint transcendentals now bypass FASC entirely
- Composed functions (tan, sinh, asin, etc.) use direct compute-tier arithmetic
- `pow(x, y) = direct_exp(direct_ln(x) * y)` — zero FASC overhead

**Test count: 1375+ across all 5 profiles, 0 failures, 0 warnings.**

---

## Next: 0.5.0 — Correctness audit + remaining composed transcendental bypass

### 1. UGOD multi-tier promotion verification

**Priority:** HIGH — correctness concern
**Effort:** ~200 lines, 1 session

Verify that UGOD overflow promotion tries at least 2 subsequent tiers before falling back to symbolic rational. Current behavior may promote to symbolic after a single tier overflow, bypassing intermediate tiers that would succeed. Either confirm the code is correct (document the guarantee) or fix it.

### 2. Complete FixedPoint direct-engine bypass for composed transcendentals

**Priority:** MEDIUM — throughput for inference consumers
**Effort:** ~300 lines, 1 session

Currently tan/asin/acos/sinh/cosh/tanh/asinh/acosh/atanh still route through FASC (`apply_unary(LazyExpr::tan)` etc.). These are FASC-composed (e.g., tan = sin/cos, sinh = (exp(x)-exp(-x))/2). Implement direct compositions using the already-direct exp/ln/sqrt/sin/cos/atan engines. Saves ~65 ns per call.

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

### n-D Clifford algebra with Vahlen matrices (L4B) — 0.5.0

Cl(n,0,1) Clifford algebra with Vahlen matrix representation. Multivector arithmetic, geometric product, inner/outer products, grade extraction, and Vahlen matrix Mobius transformations. High novelty — no zero-float Clifford algebra exists in Rust. Targeted for v0.5.0 release.

### Custom FRAC_BITS for non-realtime profiles

Extending `GMATH_FRAC_BITS` to compact (i64), embedded (i128), and higher profiles. Requires verifying that `COMPUTE_FRAC_BITS = 2 * FRAC_BITS` does not exceed the native transcendental tier's capacity. Deferred until demand.

### Public API stabilization

Pre-1.0 audit of exports, feature gating, StackValue extraction methods. Five API tiers documented (FASC, Imperative, Fused, Geometric, TQ1.9) but export hygiene not yet audited.

### Decimal sin/cos 3-stage tables

Precomputed sin/cos tables for the decimal engine. Deferred — analysis showed Taylor computation is <3% of FASC pipeline time, so tables give negligible throughput gain through the full pipeline. Revisit when decimal sin/cos becomes a bottleneck (currently not, since binary sin/cos is 8x faster and used for most inference).

---

## Non-goals

- **Floating-point interop beyond convenience**: `to_f64()`/`from_f64()` exist for user convenience. Internal float usage is architecturally forbidden.
- **Dynamic precision selection**: Profiles are compile-time. Runtime tier selection within a profile is handled by UGOD, but the base storage tier is fixed at build time.
- **GPU compute**: The library targets CPU determinism. GPU offload would compromise the cross-platform bit-identical guarantee.
