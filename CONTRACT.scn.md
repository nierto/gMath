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

@ROUNDING (deterministic, integer-only; per PATH, source-verified 2026-08-14 —
  full census: docs/design/ROUNDING_CENSUS.md):
  binary-imperative: mul=PROFILE-DEPENDENT(floor rt/compact | half-even embedded
                     | trunc-0 balanced/scientific) | div=trunc-0 ALL profiles
  binary-canonical/UGOD: mul=nearest-ties-+∞ | div=half-away
  binary downscale (all compound paths): nearest-ties-+∞
  decimal-imperative DecimalFixed: mul=banker's | div=banker's
  decimal-canonical: mul=EXACT(dp grows) | div=trunc-0
  ternary: mul=trunc-0 | div=trunc-0 | transcendentals→route-via-binary
  MEASURED: imperative vs canonical storage-tier mul differ on ~half of inexact
       products (48.7% of 44k sampled pairs, compact; 1 ulp max — floor vs
       nearest); div diverges analogously everywhere (trunc vs half-away).
       Direct storage-tier mul/div = NOT path-independent today. Compound paths
       (transcendentals, dots, decomp, matrix chains, fused) compute at tier-N+1,
       round ONCE through the shared ties-+∞ downscale → path-independent, stays.
       Unification (one rule per domain, every path: binary nearest-ties-+∞,
       decimal banker's, ternary nearest-tie-free) = ROADMAP 0.5.0 item 0c.

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
  correctness: COMPOUND results PATH-INDEPENDENT (same engines, shared downscale);
  direct storage-tier mul/div NOT yet (see @ROUNDING; unification 0.5.0/0c)

@VALIDATION: mpmath refs 50-250 digits as exact strings (no floats). Decimal paths
  CI-graded vs mootable/decimal-scaled adversarial corpus.

@LIMITS: input-representation error ONLY when a domain is pinned (0.3, 1/3 inexact
  in binary); canonical router auto-classifies + routes each literal to an exact
  domain and keeps a shadow | conditioning amplifies input error O(κ) | error is
  deterministic everywhere.
```

---

Built by **Niels Erik Toren** — [support & donations](README.md#author--support).
