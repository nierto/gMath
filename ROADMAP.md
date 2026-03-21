# gMath Roadmap

Current version: **0.1.1**

This document tracks planned work and known gaps. Items are grouped by priority, not by timeline. Nothing here is promised — this is a working list for a solo-maintained project.

---

## Next: 0.2.0

### Compact profile — Q32.32 / D32.32 / TQ16.16 / i64 symbolic

Add a `compact` deployment profile using 64-bit storage with 128-bit compute tier.

| Domain | Storage | Compute | Decimals |
| ------ | ------- | ------- | -------- |
| Binary | Q32.32 (i64) | Q64.64 (i128) | 9 |
| Decimal | D32.32 (i64) | D64.64 (i128) | 9 |
| Ternary | TQ16.16 (i64) | TQ32.32 (i128) | 9 |
| Symbolic | i64 num/den | i128 intermediate | exact |

**Why**: Q64.64 is overkill for gamedev, audio DSP, embedded control, and many real-time applications where f32 is the current default. Q32.32 with i64 storage is a native word-size operation on all modern platforms and hits the sweet spot between precision and throughput.

**What exists**: `BinaryTier2`, `DecimalTier2`, `TernaryTier2` type definitions and `checked_*` arithmetic methods already exist. UGOD dispatch, stack evaluator wiring, transcendental support, and build system profile do not.

**Scope**:

- Wire Tier 2 into UGOD dispatch for all 4 domains (add/sub/mul/div/neg)
- Add `compact` profile to build.rs and Cargo.toml
- Add `BinaryStorage = i64` / `ComputeStorage = i128` type aliases under `#[cfg(table_format = "q32_32")]`
- Transcendentals: upscale Q32.32 to Q64.64, compute with existing tables, downscale (no new table generation needed)
- ULP validation against mpmath references at Q32.32 precision
- Update README, CHANGELOG, documentation

**Estimated effort**: ~2,500 lines, primarily UGOD dispatch wiring and test infrastructure.

### Domain arithmetic fixes

Known bugs in cross-domain operations:

- Symbolic rational addition produces incorrect results in some cases
- Decimal multiplication and division have edge-case failures

These were identified during development and deferred. They need investigation and targeted fixes before any new feature work.

### Stack evaluator modularization

The stack evaluator (`src/fixed_point/universal/fasc/stack_evaluator/`) was split into submodules (mod.rs, arithmetic.rs, compute.rs, conversion.rs, domain.rs, formatting.rs, parsing.rs, transcendentals.rs, tests.rs) but the dispatch logic is still dense. Further modularization would reduce cognitive load and make the compact profile integration cleaner.

---

## Future — High Priority

### FASC LazyExpr chains for matrix operations

**Status:** DESIGN — high value, touches FASC evaluator core
**Unlocks:** Long matrix operation chains (5-20 ops without intermediate materialization)

Currently, `matrix_exp`, `matrix_sqrt`, `matrix_log`, and `matrix_pow` operate on `ComputeMatrix` at tier N+1 but materialize between operations. If a Lie group chain like `exp(A) * B * exp(C)` produces large intermediate matrices that shrink back after the full chain, the intermediate materialization causes `TierOverflow` even though the final result fits.

**Proven for scalars:** The `BinaryCompute` chain persistence already handles this for scalar transcendentals — `sin(exp(44))` succeeds via FASC LazyExpr even though `exp(44)` alone overflows Q64.64 storage (test: `ugod_chain_persistence_test.rs`).

**Proposed architecture:** `LazyMatrixExpr` — a matrix expression tree analogous to `LazyExpr`:

```
enum LazyMatrixExpr {
    Literal(FixedMatrix),
    Exp(Box<LazyMatrixExpr>),       // Padé [6/6] + scaling-squaring
    Log(Box<LazyMatrixExpr>),       // inverse scaling-squaring + Taylor
    Sqrt(Box<LazyMatrixExpr>),      // Denman-Beavers
    Mul(Box<LazyMatrixExpr>, Box<LazyMatrixExpr>),
    Add(Box<LazyMatrixExpr>, Box<LazyMatrixExpr>),
    Transpose(Box<LazyMatrixExpr>),
    Inverse(Box<LazyMatrixExpr>),
}
```

The evaluator would keep all intermediates as `ComputeMatrix` (tier N+1) and only call `to_fixed_matrix()` (downscale) at the final `evaluate_matrix()` call. This is the matrix analog of `materialize_compute()` for scalar `BinaryCompute`.

**Key constraint:** Zero heap allocation in the hot path. The `ComputeMatrix` type already uses `Vec<ComputeStorage>` (allocated once per matrix), so the chain just reuses the same allocation pattern. No new allocation model.

**Scope:** ~1000 lines (LazyMatrixExpr enum, evaluate_matrix(), operator overloading for the tree builder, integration with existing matrix_functions.rs). High risk — touches the precision-critical path. Needs dedicated session with comprehensive mpmath validation.

### Fused transcendental paths

**Status:** DESIGN — medium effort, high throughput win
**Unlocks:** Faster compound operations, fewer materialization boundaries

Common transcendental chains should have dedicated fused evaluation paths in the FASC stack evaluator, avoiding the upscale-downscale-upscale cycle between operations:

| Fused path | Operations saved | Use case |
|-----------|-----------------|----------|
| `sin_of_exp(x)` | 1 materialization | Lie group: sin(θ) where θ = exp(ω) |
| `cos_of_exp(x)` | 1 materialization | Rodrigues formula |
| `exp_of_ln(x)` | Identity → short-circuit | Roundtrip verification |
| `ln_of_exp(x)` | Identity → short-circuit | Roundtrip verification |
| `sin_cos(x)` | Share range reduction | Rodrigues: both sin(θ)/θ and (1-cos(θ))/θ² |
| `exp_of_mul(x, y)` | 1 materialization | pow(x, y) = exp(y * ln(x)) inner chain |
| `sqrt_of_sum_sq(a, b)` | 1 materialization | Norm: sqrt(a² + b²) |

**Implementation:** New `LazyExpr` variants (`SinOfExp`, `SinCos`, `ExpOfLn`, etc.) or compound opcodes in the stack evaluator. Each fused path:
1. Computes the inner operation at BinaryCompute tier (no downscale)
2. Feeds the compute-tier result directly to the outer operation
3. Materializes once at the end

**Zero allocation impact:** These are new opcodes in the existing fixed-size stack machine (`value_stack: [Option<StackValue>; 256]`). No new allocation pattern. The workspace is already sized for the most complex single operation; fused paths reuse it.

**Scope:** ~500 lines per fused path (evaluator dispatch + tests). Start with `sin_cos` (used everywhere in Rodrigues/geodesics) and `sin_of_exp`/`cos_of_exp` (Lie group hot path).

### Multi-domain matrix operations

Currently `FixedMatrix` is binary-only (`Vec<FixedPoint>` where `FixedPoint` wraps `BinaryStorage`). FASC already routes scalars per-domain (binary, decimal, ternary, symbolic). Extending this to matrices would enable:

- **Decimal matrix arithmetic** — financial-grade: interest rate matrices, transition probabilities, portfolio correlations with exact decimal storage throughout. `matrix_exp(rate_matrix * t)` where rate entries are exact decimals (0.05, 0.03).
- **Ternary matrix operations** — balanced ternary's sign-symmetric digits {-1, 0, +1} for geometric hashing. Ternary matmul preserves the balanced property.
- **Cross-domain matrix chains** — a LazyMatrixExpr chain where the input is decimal but an intermediate transcendental is computed in binary (the natural domain for sin/cos), then the result converts back to decimal for output.

Recommended approach: the `LazyMatrixExpr` from the FASC chain persistence work naturally extends to multi-domain. If the `Literal` entries carry a `StackValue` domain tag instead of raw `FixedPoint`, the chain evaluator dispatches per-domain automatically. This is approach B (domain-tagged) for bulk ops, approach A (lazy) for expression chains.

**Dependency:** FASC chain persistence first (LazyMatrixExpr + fused transcendentals), then domain extension.

### Decimal domain transcendentals

The binary domain has 0-ULP transcendentals (18 functions) via tier N+1 table-lookup + Taylor remainder. The decimal domain currently routes transcendentals through binary compute (`via-binary-compute` in the COMPONENT_STATUS_MATRIX). Native decimal transcendentals would:

- Eliminate the binary→decimal conversion round-trip (exact decimal inputs stay exact throughout)
- Enable financial-grade computations where decimal exactness matters (compound interest, actuarial, amortization)
- Support a future `g_math_finance` crate that guarantees 0-ULP decimal results for financial formulas

**Scope:** New table generation in build.rs for decimal Q-format, decimal-native exp/ln/sqrt/sin/cos at tier N+1. The algorithmic patterns are identical to binary — only the base changes (10 vs 2). Estimated ~3000 lines.

**Priority:** After FASC chain persistence (the chain architecture benefits decimal transcendentals too — no point building decimal transcendentals that materialize between operations).

### Fractal topology FASC integration

The fractal topology engine (`src/fixed_point/router/fractal_topology/`) is implemented but not wired into the FASC parse_literal() path. The engine uses 4D coordinate mapping and height-competition rules to recommend optimal domains based on input characteristics.

Integration would replace the current syntactic routing (pattern-matching on input strings) with geometric routing (domain selection based on empirical precision data). This is architecturally interesting but not urgent — the current syntactic routing works correctly.

### Ternary test coverage

Balanced ternary arithmetic is implemented with 6-tier UGOD but lacks a dedicated validation test suite comparable to the binary and decimal suites. The domain works but has not been stress-tested against reference values.

### Batch/vectorized API

The current API is scalar — one expression at a time. A batch API that processes arrays of values could genuinely benefit from AVX2 SIMD at lower tiers (8x Q32.32 values in a single 256-bit register). This would be a new API surface, not a change to the existing canonical pipeline.

### Q16.16 profile

If demand materializes for 4-decimal-digit fixed-point (retro gamedev, very constrained embedded), a `minimal` profile using Q16.16 (i32 storage, i64 compute) could be added. The pattern established by the compact profile would make this straightforward. Deferred because the value proposition at 4 decimals is thin — most users needing Q16.16 hand-roll it in 20 lines.

### Imperative geometry methods — UGOD + FASC integration

Downstream integrations required extension traits to cover geometry methods that the imperative `FixedPoint`/`FixedVector`/`FixedMatrix` types don't natively provide. These should be upstreamed into g_math as first-class UGOD-dispatched, FASC-computed methods — following the same 3-tier pattern as the binary transcendentals.

**Methods to upstream** (~300 LOC):

| Method | Type | Current impl | Target |
| ------ | ---- | ------------ | ------ |
| `square()` | FixedPoint | `self * self` | Tier 1 UGOD (inline) |
| `reciprocal()` | FixedPoint | `one() / self` | FASC BinaryCompute chain |
| `powi(i32)` | FixedPoint | binary exponentiation loop | FASC BinaryCompute chain |
| `euclidean_distance()` | FixedVector | sum-of-squares + sqrt | FASC chain (compute at Q128.128, single downscale) |
| `manhattan_distance()` | FixedVector | sum-of-abs | Tier 1 UGOD (no upscale needed) |
| `ensure_normalized()` | FixedVector | clamp 0.0–1.0 per element | Tier 1 UGOD |
| `from_f32_slice()` | FixedMatrix | loop + from_f32 | convenience constructor |
| `mul_vector()` | FixedMatrix | dot-product per row | FASC chain (accumulate at Q128.128) |

**Why FASC matters here**: `euclidean_distance()` and `mul_vector()` accumulate intermediate products. Without FASC, each multiply-then-add introduces a rounding boundary. With BinaryCompute chain persistence, the entire sum-of-squares computation stays at Q128.128 and only rounds once at materialization — identical to how transcendentals work today.

**Tier mapping** (following existing transcendental pattern):
- Tier 1 (UGOD inline): `square`, `manhattan_distance`, `ensure_normalized` — no precision benefit from upscaling
- Tier 2 (FASC BinaryCompute): `reciprocal`, `powi`, `euclidean_distance`, `mul_vector` — benefit from compute-tier intermediate precision
- Convenience: `from_f32_slice` — no arithmetic, just construction

**Estimated effort**: ~800 lines (methods + UGOD dispatch + FASC wiring + ULP validation tests against mpmath).

### Tensor decompositions (L2B)

Higher-order tensor decompositions for model compression and latent structure extraction:

- **Truncated SVD** for rank-2 tensors (reuses L1B SVD, thin wrapper)
- **HOSVD / Tucker decomposition**: unfold tensor along each mode, SVD each unfolding, reconstruct core tensor. FASC strategy: each mode-k SVD uses compute-tier dot products via existing `svd_decompose`. Core tensor contraction uses `compute_tier_dot_raw`.
- **CP decomposition via ALS** (Alternating Least Squares): iterative rank-R approximation. Each ALS step is a least-squares solve (L1C `least_squares`), convergence monitored via `convergence_threshold`.

These are optimization-tier features for distributed inference. Basic tensor contraction and SVD-based sharding work without them.

### n-D Möbius transformations / Clifford algebra (L4B)

Generalize 2D Möbius transformations to arbitrary dimensions via:

- **Vahlen matrices**: 2×2 matrices over a Clifford algebra Cl(n,0,1), acting on R^n ∪ {∞}
- **Clifford algebra representation**: geometric product, inner/outer products, versors
- Requires implementing Cl(p,q,r) multivector arithmetic in fixed-point — a substantial algebraic system

High novelty (no zero-float Clifford algebra exists in Rust). Build when conformal geometric algebra is needed for a specific application.

### Public API stabilization

Pre-1.0 cleanup: audit public exports, ensure `StackValue` methods are sufficient for all extraction needs, consider whether `FixedPoint`/`FixedVector`/`FixedMatrix` should remain public or be feature-gated.

---

## Non-goals

- **Floating-point interop beyond convenience**: `to_f64()`/`from_f64()` exist for user convenience. Internal float usage is architecturally forbidden.
- **Dynamic precision selection**: Profiles are compile-time. Runtime tier selection within a profile is handled by UGOD, but the base storage tier is fixed at build time.
- **GPU compute**: The library targets CPU determinism. GPU offload would compromise the cross-platform bit-identical guarantee.
