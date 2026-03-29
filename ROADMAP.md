# gMath Roadmap

Current version: **0.3.0**

This document tracks planned work and known gaps. Items are grouped by priority, not by timeline. Nothing here is a promise — this is a working list for a solo-maintained project.

---

## Delivered in 0.3.0

### Compact profile — Q32.32 (i64 storage, i128 compute)

`GMATH_PROFILE=compact` — 9 decimal digits, native 64-bit integer ops, 4-8x faster than Q64.64.

- `BinaryStorage = i64`, `ComputeStorage = i128` (true tier N+1)
- All 18 transcendentals via upscale→Q128.128 (I256) compute→downscale pattern
- **0 ULP** across all 90 mpmath-validated test points (dedicated + composed)
- BinaryCompute chain persistence at i128 — intermediates never downscale until output
- All 4 domains (binary, decimal, ternary, symbolic) wired for Q32.32
- 204 lib tests passing

### Realtime profile — Q16.16 (i32 storage, i64 compute)

`GMATH_PROFILE=realtime` — 4 decimal digits, native 32-bit ops, ultra-low-latency edge inference.

- `BinaryStorage = i32`, `ComputeStorage = i64` (true tier N+1)
- All 18 transcendentals via upscale→Q128.128 (I256) compute→downscale pattern
- **0 ULP** across all 90 mpmath-validated test points
- 204 lib tests passing

### TQ1.9 — Compact ternary format for inference

Standalone type for neural network weight storage. Beats fp16 precision at identical 2-byte cost.

| Property | TQ1.9 | fp16 |
|----------|-------|------|
| Storage | 2 bytes (i16) | 2 bytes |
| Precision | ~4.3 decimal digits (uniform) | ~3.3 digits (variable) |
| Range | ±1.5 | ±65504 |
| Scale | 3^9 = 19683 | N/A |

- `TritQ1_9` type with checked arithmetic (i32 intermediate)
- Balanced ternary trit decomposition/reconstruction
- UGOD promotion to TernaryTier1 on overflow
- 17 unit tests with exact integer verification

### Trit packing utilities

Public API for balanced ternary weight storage.

- `Trit` enum (`{Neg, Zero, Pos}`)
- `pack_trits` / `unpack_trits` — 5 trits/byte base-3 encoding (3^5 = 243 ≤ 255)
- 10 unit tests including 1000-trit stress roundtrip

### Compute-tier trit operations

Zero-multiply dot product and matrix-vector at tier N+1.

- `compute_tier_trit_dot_raw` — accumulate at ComputeStorage, single downscale
- `compute_tier_trit_matvec_raw` — row-wise with per-row scale factors

### Domain arithmetic fixes

Fixed 3 bugs in symbolic rational Tiny tier (i8 storage):

- `add_tiny_tier`: was ignoring second denominator, now cross-multiplies correctly
- `subtract_tiny_tier`: same fix
- `divide_tiny_tier`: was dropping sign on negative divisor

### Profile system (5 profiles)

| Profile | Storage | Compute | Bytes | Digits | Status |
|---------|---------|---------|-------|--------|--------|
| realtime | Q16.16 (i32) | Q32.32 (i64) | 4 | 4 | **0 ULP** |
| compact | Q32.32 (i64) | Q64.64 (i128) | 8 | 9 | **0 ULP** |
| embedded | Q64.64 (i128) | Q128.128 (I256) | 16 | 19 | **0 ULP** |
| balanced | Q128.128 (I256) | Q256.256 (I512) | 32 | 38 | **0 ULP** |
| scientific | Q256.256 (I512) | Q512.512 (I1024) | 64 | 77 | **0 ULP** |

All profiles use true tier N+1 computation. Transcendentals compute at two tiers above storage (tier N+1 of the compute tier) to guarantee that chain persistence introduces zero ULP at the compute tier, which vanishes completely upon downscale to storage.

---

## Delivered in 0.4.0

### LazyMatrixExpr — matrix chain persistence

`LazyMatrixExpr` enum with 14 variants: Literal, Identity, Add, Sub, Mul, ScalarMul, Transpose, Neg, Inverse, Exp, Log, Sqrt, Pow. Full operator overloading (`A * B + C`, `-A`, `A * scalar`).

All intermediates stay at `ComputeMatrix` (tier N+1). Single `to_fixed_matrix()` at `evaluate_matrix()`. Reuses existing `matrix_exp_compute`, `matrix_log_compute`, `matrix_sqrt_compute`.

- `exp(A) * B * exp(C)` — zero intermediate materializations
- `exp(log(A))` roundtrip: 0-1 ULP (chain at compute tier)
- `A^p = exp(p * log(A))` — entire chain at compute tier
- 18 integration tests, all passing

### Fused transcendental paths

- `evaluate_sincos(&LazyExpr)` — single range reduction for both sin and cos
- Identity short-circuits: `exp(ln(x))` → x, `ln(exp(x))` → x (pattern match in evaluator)
- 12 tests including Pythagorean identity, large-angle, compound expressions

### Multi-domain matrix operations

`DomainMatrix` type with `Vec<StackValue>` entries. Each element carries its own domain tag (Binary, Decimal, Ternary, Symbolic).

- Same-domain operations use native dispatch
- Cross-domain operations route through rational automatically
- `to_fixed_matrix()` converts any domain to binary via evaluator
- Decimal matrix arithmetic: 0-ULP exact for financial-grade operations
- Cross-domain matmul: Binary * Decimal routes correctly
- 17 tests covering all 4 domains + cross-domain + trace

### Fused compute-tier operations

`fused` module with 5 operations that keep ALL intermediates at tier N+1 (single downscale at end):

| Function | Eliminated materializations | Primary consumer |
|----------|---------------------------|-----------------|
| `sqrt_sum_sq(&[FixedPoint])` | 1 per norm call | Distance/norm in metric spaces |
| `euclidean_distance(&[FixedPoint], &[FixedPoint])` | 2 per distance call | Nearest-neighbor, geodesic distance |
| `softmax(&[FixedPoint])` | seq_len per head | Attention weight normalization |
| `rms_norm_factor(&[FixedPoint], eps)` | 3 per layer | Per-layer normalization |
| `silu(FixedPoint)` | 1 per activation | SwiGLU gate activation |

Also added `FixedVector::length_fused()` and `FixedVector::distance_to()` convenience methods.

- 30 tests, all passing (mpmath-validated)
- `exp_at_compute_tier()` free function added to compute.rs for fused paths

### Bug fixes

- Fixed 5 incorrect mpmath reference values in trace test files (sqrt(3) constants were wrong by millions of ULP — algorithm was correct, expected values were bogus)

---

## Next: 0.3.89 — TQ1.9 inference layer

Dedicated TQ1.9 module decoupled from FASC routing/shadow/domain machinery. Optimized for throughput-critical ternary workloads (~196 matvec calls per token for typical 3B-parameter models). Standalone module — no close coupling with gMath internals where that would affect speed.

### 1. Dedicated inference module (`inference/`)

**Priority:** HIGH — architectural prerequisite for everything below
**Estimated effort:** ~300 lines, 1 session

New top-level module `src/fixed_point/inference/` containing:
- `TQ19Matrix` type (row-major `Vec<i16>`, rows/cols metadata)
- `tq19_matvec()` — compute-tier accumulation, single `/19683` per row
- `tq19_matvec_batch()` — multiple input vectors, cache-friendly
- Re-export trit packing utilities
- No StackValue, no CompactShadow, no FASC evaluator in the hot path
- Direct `i16 × BinaryStorage → ComputeStorage` arithmetic

### 2. AVX2 SIMD inner loop

**Priority:** HIGH — largest single throughput gain
**Estimated effort:** ~400 lines, 1-2 sessions

Zero-multiply SIMD for trit dot product:
- Compact profile (i64 activations): **4× i64 add/sub per AVX2 cycle** via `_mm256_add_epi64` / `_mm256_sub_epi64` with trit-derived masks
- Realtime profile (i32 activations): **8× i32 per cycle**
- Pre-decode trits into sign masks (+1/0/-1 → add/skip/sub) outside the SIMD loop
- Feature-gated: `#[cfg(target_feature = "avx2")]` with scalar fallback
- Each CPU core has its own AVX2 unit — fully parallelizable with rayon, no shared bottleneck

### 3. Row-parallel rayon dispatch

**Priority:** HIGH — proven pattern, upstreamed into gMath
**Estimated effort:** ~100 lines, 1 session

`tq19_matvec_par()` with `rayon::par_iter` over rows. Each row is independent (embarrassingly parallel). Consumers get row-level parallelism for free.

### 4. Batch matvec

**Priority:** MEDIUM — cache efficiency for multi-token inference
**Estimated effort:** ~200 lines, 1 session

`tq19_matvec_batch(matrix, vectors: &[&[BinaryStorage]]) -> Vec<Vec<BinaryStorage>>` — process N input vectors against the same weight matrix. Benefits:
- Weight matrix stays in L2/L3 cache across vectors (row data loaded once, applied N times)
- Amortizes trit decoding cost across batch
- Enables prefill-phase parallelism (multiple tokens in flight)

### 5. Pre-decoded trit table

**Priority:** MEDIUM — eliminates repeated unpack overhead
**Estimated effort:** ~150 lines, 1 session

Precompute 256-entry lookup table mapping each byte to 5 sign values ({-1, 0, +1}). Const version compiled into `.rodata` — avoids per-element divmod in the inner loop.

---

## Next: 0.4.0 — remaining items

### 1. Decimal domain transcendentals

**Priority:** HIGH — enables true multi-domain composability
**Estimated effort:** ~3,000 lines, 3-4 sessions

Native decimal exp/ln/sqrt/sin/cos/tan/atan at tier N+1 with decimal-specific tables. Currently decimal transcendentals route through binary compute, which works but introduces a domain conversion round-trip. Native decimal transcendentals mean:
- All composed methods (sinh, cosh, tanh, asin, acos, pow, etc.) become multi-domain automatically — they're built from the 5 dedicated algorithms which each domain implements natively
- `DomainMatrix` operations with decimal entries get native transcendentals without binary conversion
- Financial/accounting domains get exact decimal exp/ln for compound interest, Black-Scholes, etc.
- FASC chain persistence works per-domain: decimal intermediates stay decimal throughout

Requires: decimal-specific table generation in `build.rs`, decimal tier N+1 compute infrastructure, decimal BinaryCompute equivalent.

### 2. UGOD multi-tier promotion

**Priority:** HIGH — correctness concern
**Estimated effort:** ~200 lines, 1 session

Verify and enforce that UGOD overflow promotion tries at least 2 subsequent tiers before falling back to symbolic rational. Current behavior may fall back to symbolic after a single tier overflow, which is wasteful when the next tier would succeed.

### 3. Stack evaluator modularization

**Priority:** MEDIUM — reduces cognitive load for future changes
**Estimated effort:** ~750 lines, 2 sessions

Extract `profile_dispatch!` macro to replace the 5-way `#[cfg(table_format)]` blocks that now exist across all submodules. Each function currently has 5 copy-paste cfg arms — a macro would reduce this to 1 invocation per function.

---

## Future — High Priority

### Fractal topology FASC integration

Wire the fractal topology engine into FASC `parse_literal()` for geometric domain routing. Replace syntactic routing with data-driven routing. Not urgent — current routing works correctly.

### Ternary test coverage

Balanced ternary arithmetic lacks dedicated validation suite. The domain works but needs stress-testing against reference values.

### Batch/vectorized API

SIMD-friendly array processing beyond TQ1.9. Especially compelling with Q32.32 (8x i32 in AVX2 register) and Q16.16 (16x i16 in AVX2). New API surface.

### Imperative geometry methods — UGOD + FASC integration

Upstream `square()`, `reciprocal()`, `powi()`, `euclidean_distance()`, `manhattan_distance()`, `mul_vector()` etc. as first-class UGOD-dispatched, FASC-computed methods. ~800 lines.

### Tensor decompositions (L2B)

Truncated SVD, HOSVD/Tucker, CP/ALS for model compression. Optimization-tier features for distributed inference.

### n-D Möbius transformations / Clifford algebra (L4B)

Vahlen matrices over Cl(n,0,1). High novelty — no zero-float Clifford algebra in Rust.

### Public API stabilization

Pre-1.0 audit of exports, feature gating, StackValue extraction methods.

---

## Non-goals

- **Floating-point interop beyond convenience**: `to_f64()`/`from_f64()` exist for user convenience. Internal float usage is architecturally forbidden.
- **Dynamic precision selection**: Profiles are compile-time. Runtime tier selection within a profile is handled by UGOD, but the base storage tier is fixed at build time.
- **GPU compute**: The library targets CPU determinism. GPU offload would compromise the cross-platform bit-identical guarantee.
