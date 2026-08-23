# CONTRACT.scn — g_math agent primer

Condensed prime for an agent building against `g_math`. Authoritative prose:
[CONTRACT.md](CONTRACT.md). Per-symbol index: [PUBLIC_API.md](PUBLIC_API.md).
Code wins over docs.

```yaml
@CRATE: g_math (Rust, edition-2021, license MIT-OR-Apache-2.0)
  role: multi-domain fixed-point arithmetic, zero-float, deterministic
  default-runtime-deps: NONE (pure compute lib; no I/O, no network)

@SURFACE (supported; everything else internal, may-change):
  canonical:   g_math::canonical{gmath, gmath_parse, evaluate, evaluate_sincos,
               evaluate_sinhcosh, evaluate_matrix, set_gmath_mode, reset_gmath_mode,
               LazyExpr, LazyMatrixExpr, DomainMatrix}
  imperative:  g_math::fixed_point{FixedPoint, FixedVector, FixedMatrix}
               + DecimalFixed<DECIMALS>
  fused:       g_math::fixed_point::imperative::fused{sqrt_sum_sq, euclidean_distance,
               softmax, softmax_mix, rms_norm_factor, silu}
  geometric:   imperative::{decompose, derived, matrix_functions, manifold, lie_group,
               curvature, projective, fiber_bundle, ode, tensor, tensor_decompose,
               serialization}
  tq19:        g_math::tq19 (feature=inference){TQ19Matrix, PlanarTQ19, HybridTQ19,
               tq19_dot, ...}
  internal:    universal/fasc, domain impls, I256/I512/I1024/I2048, shadow/router

@DETERMINISM (core guarantee):
  fixed-profile + identical-input → bit-identical-output ∀ platform/arch
  mechanism: FLOAT_BAN — no f32/f64 in arithmetic|validation|comparison
  exception: from_f64/to_f64 = caller-convenience ONLY, never internal
  use-cases: consensus, financial-audit, reproducible-science

@ROUNDING (deterministic, integer-only; UNIFIED 0.5.0 — one rule per domain,
  every path; gate: tests/rounding_unification.rs; evidence:
  docs/design/ROUNDING_CENSUS.md):
  binary:  nearest-ties-+∞ EVERYWHERE (mul, div, downscales, decimal→binary
           coercion; all profiles, imperative == canonical == fused)
  decimal: exact-when-representable; banker's where rounding occurs
           (canonical mul = dp-growth exact; canonical div t1-5 =
           exact-or-rational-fallback; DecimalFixed + t6 div = banker's)
  ternary: nearest — tie-free mul/div3 (odd scale); ties-+∞ at div and
           conversion-in (0.5 → 3281, -0.5 → -3280: documented asymmetry)
  EXCEPTION (contracted): tq19 matvec_q2f narrowing stays truncation
           (0.4.31 bit-reproducibility contract).
  Compound paths: tier-N+1, round ONCE at downscale (unchanged).

@TIER-N+1: all transcendentals + accumulations compute one width above storage,
  single downscale at end. Chains (canonical layer) keep intermediates wide until
  materialization → chain rounds once, not per-step.

@PROFILES (compile-time, GMATH_PROFILE or cargo feature):
  realtime Q16.16/i32(compute i64, ~4dp; GMATH_FRAC_BITS 2..30 custom split)
  compact  Q32.32/i64(i128, ~9dp)
  embedded Q64.64/i128(I256, ~19dp) [DEFAULT]
  balanced Q128.128/I256(I512, ~38dp)
  scientific Q256.256/I512(I1024, ~77dp)
  SWITCH: rm -rf target/*/incremental BETWEEN profiles (stale cfg → crash)

@CROSS-PROFILE:
  guaranteed: determinism WITHIN a profile
  NOT-yet-contracted: widening-exact / single-rounding-narrowing / foreign-profile
    serialization semantics — planned (ROADMAP U2). Do NOT rely on cross-profile
    bit relationships yet.

@FEATURES: infinite-precision(num-bigint symbolic tier) | serde | inference(rayon
  + tq19) | rebuild-tables | {realtime|compact|embedded|balanced|scientific} |
  legacy-tests. Core (all domains, transcendentals, wide ints, UGOD) always compiled.

@PATH-CHOICE:
  canonical → mixed/chained/user-input work (routing + chain persistence, overhead)
  imperative → known-domain hot loops (direct, Copy, no routing)
  fused → ML accumulation shapes (softmax/silu/rms/distance, compute-tier)
  tq19 → ternary-quantized inference (feature=inference)
  correctness: PATH-INDEPENDENT — compound results (same engines, shared
  downscale) AND direct storage-tier arithmetic (unified 0.5.0, test-gated)

@VALIDATION: mpmath refs 50-250 digits as exact strings (no floats). Decimal paths
  CI-graded vs mootable/decimal-scaled adversarial corpus.

@LIMITS: input-representation error ONLY when a domain is pinned (0.3, 1/3 inexact
  in binary); canonical router auto-classifies + routes each literal to an exact
  domain and keeps a shadow | conditioning amplifies input error O(κ) | error is
  deterministic everywhere.

@OVERFLOW(0.5.0): canonical arithmetic on representable inputs = EXACT (tier
  promotion → exact-rational fallback) OR loud typed error. NEVER silent wrap —
  any tier, any domain, coercions + literal parsing included (out-of-range
  literals on narrow profiles parse into exact symbolic). Gate:
  tests/ugod_promotion_validation.rs per profile.
```

---

Built by **Niels Erik Toren** — [support & donations](README.md#author--support).
