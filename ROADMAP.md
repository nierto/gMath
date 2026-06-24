# gMath Roadmap

Current version: **0.4.24**

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

---

## Next: 0.5.0 — Correctness audit + remaining composed transcendental bypass

### 1. UGOD multi-tier promotion verification

**Priority:** HIGH — correctness concern
**Effort:** ~200 lines, 1 session

Verify that UGOD overflow promotion tries at least 2 subsequent tiers before falling back to symbolic rational. Current behavior may promote to symbolic after a single tier overflow, bypassing intermediate tiers that would succeed. Either confirm the code is correct (document the guarantee) or fix it.

### 2. Complete FixedPoint direct-engine bypass for composed transcendentals

**Priority:** MEDIUM — throughput for inference consumers
**Effort:** ~300 lines, 1 session

The infallible composed methods (tan/asin/acos/sinh/cosh/tanh/asinh/acosh/atanh) are already direct, but their fallible `try_*` variants still route through FASC (`try_apply_unary(LazyExpr::tan)` etc.). Implement direct compositions for the `try_*` variants using the already-direct exp/ln/sqrt/sin/cos/atan engines. Avoids the FASC pipeline overhead per call.

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
