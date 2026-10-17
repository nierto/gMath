# gMath integration contract

The precise, normative statement of what `g_math` guarantees to a caller: the
supported API surface and its stability boundary, the determinism guarantee, the
rounding rules, cross-format semantics, and what the crate requires. The
[README](README.md) orients; the [guides](README.md#guides) explain each layer;
this document is the authority. Where this document and the running code disagree,
the code is authoritative: fix this document.

Agents: prime from **[CONTRACT.scn.md](CONTRACT.scn.md)**.

---

## 1. Public API surface & stability boundary

The supported, build-with surface is:

- **Canonical**: `g_math::canonical` (`gmath`, `gmath_parse`, `evaluate`,
  `evaluate_sincos`, `evaluate_sinhcosh`, `evaluate_matrix`, `set_gmath_mode`,
  `reset_gmath_mode`, and the `LazyExpr` / `LazyMatrixExpr` / `DomainMatrix` types).
- **Formats**: `g_math::fixed_point::{Fixed, FixedNum, Q16_16, Q32_32, Q64_64,
  Q128_128, Q256_256, Compute, Storage, WideInt, CircleStorage, Scalar,
  I256, I512, I1024, I2048}`: `Fixed<S, F>` for `S = i32` with `2 <= F <= 30`,
  `i64` with 32, `i128` with 64, `I256` with 128, `I512` with 256.
- **Imperative**: `Fixed<S, F>`, `g_math::fixed_point::{FixedVector, FixedMatrix,
  Interval}` and `DecimalFixed<DECIMALS>`.
- **Fused**: `g_math::fixed_point::imperative::fused`.
- **Geometric**: `g_math::fixed_point::imperative::{decompose, derived,
  matrix_functions, manifold, lie_group, curvature, projective, fiber_bundle, ode,
  tensor, tensor_decompose, serialization}`.
- **Compute tier and wide tier**: `g_math::compute_tier`, `g_math::wide`.

The ternary inference kernels (TQ1.9) were `g_math::tq19` behind the
`inference` feature up to 0.6.8; they are the separate crate
`g_math_inference` from 0.7.0, which relies only on the surface above.

Everything else (the `universal`/`fasc` internals, the domain implementations
beneath `DecimalFixed` and the ternary types, the shadow/router internals, and
every `#[doc(hidden)]` item) is **internal and may change
between releases**. The complete per-symbol index of the supported surface is
generated into **[PUBLIC_API.md](PUBLIC_API.md)**; it is not hand-maintained.

This crate is pre-1.0: the surface above is what stabilization targets, but
breaking changes may still occur on minor versions until 1.0. Version lives in
`Cargo.toml`.

## 2. Determinism guarantee

**For a given format, identical input produces bit-identical output on every
platform and architecture.** This is the core guarantee and the reason the crate
exists; it makes results suitable for blockchain consensus, financial auditing,
and reproducible scientific computation.

The mechanism is a hard constraint: no `f32`/`f64` appears anywhere in the
arithmetic, validation, or comparison paths. Float conversions exist **only** as
caller-convenience wrappers (`from_f64` / `to_f64` and their f32 twins) and are
never used for internal correctness. They are themselves integer operations on the
IEEE bits: `to_f64` / `to_f32` are exact when the raw value fits the significand
and round to nearest with ties to even otherwise; `from_f64` / `from_f32`
truncate toward zero and reject NaN, infinity and out-of-range values (an error
from the `try_` forms, a panic otherwise, never a wrapped value). All rounding is
integer arithmetic, so the result (including any error) is identical everywhere.

## 3. Rounding contract

**Unified 2026-08-23 (0.5.0)**: one rule per domain, identical on every
path (imperative, canonical/UGOD, fused, coercions), gated permanently by
`tests/rounding_unification.rs` (cross-path bit-equality sweeps including
constructed exact-tie inputs, at every format):

| Domain | Rule (everywhere) | Notes |
| ------ | ----------------- | ----- |
| Binary | round-to-nearest, ties toward +∞ | multiply, divide, every wide-tier downscale, and decimal→binary coercion; one rule, every format |
| Decimal | exact when representable; banker's (half-even) where rounding occurs | canonical multiply grows decimal places (no rounding); canonical divide tiers 1–5 are exact-or-rational-fallback (`PrecisionLoss` → symbolic) and never round; `DecimalFixed<D>` and the tier-6 best-effort divide round banker's; compute-tier intermediates inside the decimal transcendentals and canonical decimal-compute chains carry guard digits and round half away from zero before the one half-even narrowing. `DecimalFixed` division is one rounding from the exact quotient, and `mul_div` one rounding from the exact `a * b / c`. Since 0.6.5 the `_with` methods take the tie rule explicitly (`DecimalRounding::HalfEven` or `HalfUp`, away from zero); the default everywhere is half-even |
| Balanced ternary | round-to-nearest | tie-free for multiply and `div3` (odd scale, contract theorem); ties toward +∞ where ties exist (divide, conversion in: e.g. `0.5` → raw 29525, `-0.5` → raw −29524 at TQ10.10, the documented +∞ tie asymmetry) |

Exactness-first remains the prior rule everywhere: a result representable
in its domain is returned exactly; rounding fires only at the single
narrowing a path performs. Compound paths (transcendentals, dots,
decompositions, chains, fused ops) compute at tier N+1 and round exactly
once.

**Correct rounding (0.7.0).** These results are the correctly rounded value
of the exact mathematical result (binary: nearest, ties toward +∞; decimal:
nearest, ties to even), at every format and over the whole domain and range:
the 17 transcendentals on `Fixed` and on single-function canonical trees;
every canonical tree with an operation (rounded once from the exact value of
the whole tree, whose leaves are exact: decimal literals as their rational
value, named constants as the real number); the decimal transcendentals of
`DecimalFixed<D>`; and the closed-form geometry maps (Sphere and
HyperbolicSpace distance, exp, log and transport; SO(3) and SE(3) exp, log,
exp_map, log_map and distance; SO(3) transport). A fast stage decides when
its result is farther from a rounding boundary than its error bound; the
bounds of the fast engines are measured (calibrated per format with a
factor of headroom and stress-tested), not proven, and the sure stage at
128 to 4160 bits carries derived bounds. A value that is exactly a tie and
whose exactness cannot be shown within that precision is rounded as a tie.
Gates (mpmath, no tolerance): `tests/exp_full_range_validation.rs`,
`tests/chain_correct_rounding_validation.rs`,
`tests/decimal_chain_correct_rounding_validation.rs`,
`tests/geometry_correct_rounding_validation.rs`,
`tests/decimal_tier_by_d_validation.rs`. Iterative and integrating
operations (decompositions, matrix functions, ODE, curvature) are within
one unit of their mpmath reference, not proven correctly rounded.

**"Certified" is a technical term, not a warranty**: it means the interval
provably contains the exact result of the stated integer computation on the
values passed in, or that a verdict is proven for the matrix passed in;
the software is provided as is under its licenses, and the README's
disclaimer applies to every certified operation.

**Certified enclosures round outward, by design (0.6.0).**
`Interval` and `DecimalInterval<D>` are the one place a result is not
rounded to nearest: the lower endpoint rounds toward −∞ and the upper toward
+∞ at the single narrowing of each compound operation, because that
directed rounding is what makes the bracket sound. The scalar paths are
untouched: the same exact compute-tier accumulation feeds `nearest` for a
scalar and `floor`/`ceil` for an interval, so the scalar always lies inside
its interval, and the table above still holds on every scalar path. The
quadratic form is the worked case (0.6.1): `fused::quadratic_form` narrows
ONE exact value (every `v_i m_ij v_j` an exact triple product at three times
the fractional bits) to nearest with ties toward +∞, and
`Interval::quadratic_form` narrows the same value to floor and ceil, so the
scalar is correctly rounded and its bracket is at most 1 ulp wide. One
rounding per expression, not per stage, is what the "round once" rule means
for a compound operation. The
exact predicates (`orient2d`, `orient3d`, `incircle`, `insphere`,
`pd_verdict`) never round; they evaluate integer determinants exactly or
run the interval Cholesky. The square root is the exact integer root on
words up to 256 bits, and at Q256.256 the engine computes an exact floor at
tier N+1 (certified by an integer check) before
the single nearest downscale, which is the rule above applied, not an
exception to it. Gates: `tests/interval_enclosure.rs`,
`tests/decimal_interval_enclosure.rs`, `tests/certified_geometry_refs_validation.rs`.

History: before unification the rules differed by operation, by path, and
(imperatively) by build profile: direct storage-tier multiply diverged between
paths on ~half of inexact products (measured 48.7% of 44k pairs, 1 ulp).
The full per-site before/after evidence is preserved in
[docs/design/ROUNDING_CENSUS.md](docs/design/ROUNDING_CENSUS.md).

## 4. Cross-format semantics

Every format is a distinct type, and any number of them coexist in one
build and one binary. Different formats have different bit-widths and
therefore round to different values; a Q32.32 result cannot hold Q64.64
digits. That divergence is inherent.

- **Guaranteed**: determinism *within* a format (§2), and correct rounding
  at each format for the operations listed in §3.
- **Not yet contracted**: an exact conversion between formats (widening as a
  pure bit shift, narrowing as one correct rounding). Convert through a
  value's decimal string or its raw integer meanwhile; this is a tracked
  roadmap item.

Serialization (`imperative::serialization`) tags each scalar with its word
and split (`[0x20 | code, F as u16 big-endian]`); `from_bytes` rejects bytes
of another format with an error rather than reinterpreting them, and
`from_bytes_legacy` reads the one-byte tags 0.6.x wrote.

## 5. Precision & limits

Accuracy is defined by the test suite against mpmath references (50–250 digits,
embedded as exact strings), not by adjectives. The methodology and the honest
limits are documented in [the precision guide](docs/README_PRECISION.md):
input-representation error for values inexact in a pinned domain (the canonical
router auto-routes each literal to a domain where it is exact), condition-number
amplification for ill-conditioned systems, and the determinism of whatever error
remains. Both user-reachable decimal paths
(imperative `DecimalFixed` and canonical `gmath()`) are graded in CI against
[mootable/decimal-scaled](https://github.com/mootable/decimal-scaled)'s independent
adversarial corpus.

Overflow behavior (0.5.0): canonical arithmetic on representable inputs either
returns the exact value (promoting through wider tiers or falling back to the
exact rational domain) or fails loud with a typed error. It never silently
wraps, at any tier, in any domain, including domain coercions and literal
parsing (values beyond a narrow format's binary or decimal range parse into
the exact symbolic domain). Gated at every format by
`tests/ugod_promotion_validation.rs`.

Imperative overflow (0.6.4): the `Fixed` and `DecimalFixed` operators
(`+ - * /`, unary `-`), conversions (`from_int`, `to_int`, `from_integer`,
`from_parts`), transcendentals, and every narrowing inside the library either
return the in-range result or panic; each has a `try_` form returning
`Err(TierOverflow)`, `Err(DivisionByZero)` or `Err(DomainError)` (and, on
`DecimalFixed`, `Err(PrecisionLimit)` for an in-domain argument that rounds
onto a domain boundary at the compute precision), which never panics. None
wraps or saturates (0.7.0 extends this to the inverse trigonometric and
hyperbolic functions outside their domain and to the wide balanced-ternary
add and subtract). The fixed-width types cannot promote; for promotion use
the canonical layer. Gated by `tests/operator_overflow_validation.rs` (every
format, the 32-bit splits 2 to 30), `tests/narrowing_defects_validation.rs` and
`tests/decimal_try_transcendentals_validation.rs`.

## 6. Requirements & dependencies

- **Edition** 2021. No MSRV is currently pinned.
- **Default build has zero runtime dependencies.** Lookup tables are generated at
  build time by a pure-Rust `build.rs` (build-dependencies only) and are checked
  in, so the default build does not regenerate them.
- **Optional runtime dependencies**, each behind a feature: `infinite-precision`
  (`num-bigint`/`num-traits`/`num-integer`, for the BigInt symbolic tier) and
  `serde` (`serde`).
- **No build-time configuration.** Every format is compiled into every build;
  `build.rs` reads no environment (0.6.x selected one format through
  `GMATH_PROFILE`).

## 7. What gMath consumes

Nothing at runtime by default; it is a pure computation library with no I/O, no
network, and no ambient state beyond a thread-local evaluator workspace per
format and the thread-local constant caches of the sure path. It
produces values; it does not call out.

## Disclaimer

This software is provided **"as is"**, without warranty of any kind, express or
implied. Use of this software is entirely at your own risk. In no event shall the
author or contributors be held liable for any damages arising from the use or
inability to use this software.

---

Built by **Niels Erik Toren** · [support & donations](README.md#author--support).
