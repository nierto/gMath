# CONTRACT.scn: g_math agent primer

Condensed prime for an agent building against `g_math`. Authoritative prose:
[CONTRACT.md](CONTRACT.md). Per-symbol index: [PUBLIC_API.md](PUBLIC_API.md).
Code wins over docs.

```yaml
@CRATE: g_math (Rust, edition-2021, license MIT-OR-Apache-2.0)
  role: multi-domain fixed-point arithmetic, zero-float, deterministic
  default-runtime-deps: NONE (pure compute lib; no I/O, no network)

@SURFACE (supported; everything else internal, may-change):
  formats:     g_math::fixed_point{Fixed<S, F>, FixedNum, Q16_16, Q32_32, Q64_64,
               Q128_128, Q256_256, Compute<T>, Storage, WideInt, CircleStorage,
               Scalar, I256, I512, I1024, I2048}
               S=i32 (2<=F<=30) | i64 (32) | i128 (64) | I256 (128) | I512 (256)
  canonical:   g_math::canonical{gmath, gmath_parse, evaluate, evaluate_sincos,
               evaluate_sinhcosh, evaluate_matrix, set_gmath_mode, reset_gmath_mode,
               LazyExpr<T>, LazyMatrixExpr<T>, DomainMatrix<T>}
               usage: evaluate::<Q64_64>(&(gmath("1.5") + gmath("2.5")))
  imperative:  Fixed<S, F>, FixedVector<T>, FixedMatrix<T>, Interval<T>
               + DecimalFixed<DECIMALS> (word chosen by D: Dp<D>: DecimalTier)
  fused:       g_math::fixed_point::imperative::fused{sqrt_sum_sq, euclidean_distance,
               softmax, softmax_mix(_flat/_values), rms_norm(_factor), silu,
               sigmoid_mul, dot_many, entropy, quadratic_form, ...}
  geometric:   imperative::{decompose, derived, matrix_functions, manifold, lie_group,
               curvature, projective, fiber_bundle, ode, tensor, tensor_decompose,
               serialization}
  tiers:       g_math::compute_tier (generic, ungated), g_math::wide (Q64.64)
  moved-out:   TQ1.9 inference (was g_math::tq19 + feature inference to 0.6.8)
               = separate crate g_math_inference (0.7.0+)
  internal:    universal/fasc, domain impls, shadow/router, #[doc(hidden)] items

@DETERMINISM (core guarantee):
  fixed-format + identical-input → bit-identical-output ∀ platform/arch
  mechanism: FLOAT_BAN, no f32/f64 in arithmetic|validation|comparison
  exception: from_f64/to_f64 (+ f32 twins) = caller-convenience ONLY, never internal
  float-boundary: to_f64|to_f32 := IEEE bits assembled from raw (integer ops) → exact if raw fits significand, else nearest-even
                  from_f64|from_f32 := truncate toward zero; NaN → InvalidInput, inf|out-of-range → TierOverflow (try_), panic (infallible), never wrap
                  gate: tests/float_boundary_validation.rs
  use-cases: consensus, financial-audit, reproducible-science

@ROUNDING (deterministic, integer-only; UNIFIED 0.5.0; one rule per domain,
  every path; gate: tests/rounding_unification.rs; evidence:
  docs/design/ROUNDING_CENSUS.md):
  binary:  nearest-ties-+∞ EVERYWHERE (mul, div, downscales, decimal→binary
           coercion; every format, imperative == canonical == fused)
  decimal: exact-when-representable; banker's where rounding occurs
           (canonical mul = dp-growth exact; canonical div t1-5 =
           exact-or-rational-fallback; DecimalFixed + t6 div = banker's)
  ternary: nearest: tie-free mul/div3 (odd scale); ties-+∞ at div and
           conversion-in (0.5 → 29525, -0.5 → -29524 @TQ10.10: documented asymmetry)
  Compound paths: tier-N+1, round ONCE at downscale.
  CORRECTLY-ROUNDED (0.7.0, mpmath gates, no tolerance): 17 transcendentals
           (full domain+range, every format) | canonical trees (one rounding
           from the exact value of the whole tree) | DecimalFixed
           transcendentals | closed-form geometry (Sphere/Hyperbolic maps,
           SO(3)/SE(3) exp/log/maps/distance). Method: fast stage + sure
           stage 128..4160 bits (stack-only); fast bounds MEASURED not proven.
           Decompositions/matrix functions/ODE/curvature: <= 1 unit, not
           correctly rounded.

@TIER-N+1: all transcendentals + accumulations compute one width above storage;
  canonical chains keep intermediates wide, then round once from the exact value.

@FORMATS (0.7.0: a type per value, NO build-time profile):
  Fixed<i32, F> any 2..=30 (Q16_16, Q22.10 = Fixed<i32,10>, ...; compute i64)
  Q32_32 (i128) | Q64_64 (I256) | Q128_128 (I512) | Q256_256 (I1024)
  every format compiled in every build; any number per program/dep-graph
  removed: GMATH_PROFILE, GMATH_FRAC_BITS, profile features, FixedPoint alias
  migration: docs/UPGRADING_0.7.md

@CROSS-FORMAT:
  guaranteed: determinism WITHIN a format; correct rounding per format (above)
  NOT-yet-contracted: exact widening / single-rounding narrowing between
    formats (ROADMAP U2). Serialization tags word+split; from_bytes refuses a
    foreign format (Err); from_bytes_legacy reads 0.6.x one-byte tags.

@FEATURES: infinite-precision(num-bigint symbolic tier) | serde | rebuild-tables.
  No per-format features. Core (every format, domain, transcendental, wide int,
  UGOD) always compiled.

@PATH-CHOICE:
  canonical → mixed/chained/user-input work (routing, one rounding per tree)
  imperative → known-domain hot loops (direct, Copy, no routing)
  fused → ML accumulation shapes (softmax/silu/rms/distance, compute-tier)
  correctness: PATH-INDEPENDENT for single functions (both correctly rounded)
  AND direct storage-tier arithmetic (unified 0.5.0, test-gated)

@VALIDATION: mpmath refs 50-250 digits as exact strings (no floats). Decimal paths
  CI-graded vs mootable/decimal-scaled adversarial corpus.

@LIMITS: input-representation error ONLY when a domain is pinned (0.3, 1/3 inexact
  in binary); canonical router auto-classifies + routes each literal to an exact
  domain and keeps a shadow | conditioning amplifies input error O(κ) | error is
  deterministic everywhere.

@OVERFLOW(0.5.0): canonical arithmetic on representable inputs = EXACT (tier
  promotion → exact-rational fallback) OR loud typed error. NEVER silent wrap:
  any tier, any domain, coercions + literal parsing included (out-of-range
  literals in narrow formats parse into exact symbolic). Gate:
  tests/ugod_promotion_validation.rs at every format.
@OVERFLOW_IMPERATIVE(0.6.4): Fixed|DecimalFixed operators (+ - * / neg),
  from_int|to_int|from_integer|from_parts, every internal narrowing = in-range
  result OR panic; try_ twin → Err(TierOverflow|DivisionByZero). NEVER wrap,
  NEVER saturate (0.7.0: also asin/acos/atanh/acosh out of domain, ternary
  tq80/tq320 add/sub). No promotion (fixed width): canonical layer for UGOD. Gates:
  tests/operator_overflow_validation.rs, tests/narrowing_defects_validation.rs.
```

---

Built by **Niels Erik Toren** · [support & donations](README.md#author--support).
