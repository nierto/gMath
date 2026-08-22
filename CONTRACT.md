# gMath integration contract

The precise, normative statement of what `g_math` guarantees to a caller: the
supported API surface and its stability boundary, the determinism guarantee, the
rounding rules, cross-profile semantics, and what the crate requires. The
[README](README.md) orients; the [guides](README.md#guides) explain each layer;
this document is the authority. Where this document and the running code disagree,
the code is authoritative — fix this document.

Agents: prime from **[CONTRACT.scn.md](CONTRACT.scn.md)**.

---

## 1. Public API surface & stability boundary

The supported, build-with surface is:

- **Canonical** — `g_math::canonical` (`gmath`, `gmath_parse`, `evaluate`,
  `evaluate_sincos`, `evaluate_sinhcosh`, `evaluate_matrix`, `set_gmath_mode`,
  `reset_gmath_mode`, and the `LazyExpr` / `LazyMatrixExpr` / `DomainMatrix` types).
- **Imperative** — `g_math::fixed_point::{FixedPoint, FixedVector, FixedMatrix}`
  and `DecimalFixed<DECIMALS>`.
- **Fused** — `g_math::fixed_point::imperative::fused`.
- **Geometric** — `g_math::fixed_point::imperative::{decompose, derived,
  matrix_functions, manifold, lie_group, curvature, projective, fiber_bundle, ode,
  tensor, tensor_decompose, serialization}`.
- **TQ1.9** — `g_math::tq19` (feature `inference`).

Everything else — the `universal`/`fasc` internals, the domain implementations
beneath `DecimalFixed` and the ternary types, wide-integer types (`I256`/`I512`/
`I1024`/`I2048`), the shadow/router internals — is **internal and may change
between releases**. The complete per-symbol index of the supported surface is
generated into **[PUBLIC_API.md](PUBLIC_API.md)**; it is not hand-maintained.

This crate is pre-1.0: the surface above is what stabilization targets, but
breaking changes may still occur on minor versions until 1.0. Version lives in
`Cargo.toml`.

## 2. Determinism guarantee

**For a fixed profile, identical input produces bit-identical output on every
platform and architecture.** This is the core guarantee and the reason the crate
exists — it makes results suitable for blockchain consensus, financial auditing,
and reproducible scientific computation.

The mechanism is a hard constraint: no `f32`/`f64` appears anywhere in the
arithmetic, validation, or comparison paths. Float conversions exist **only** as
caller-convenience wrappers (`from_f64` / `to_f64`) and are never used for internal
correctness. All rounding is integer arithmetic, so the result — including any
error — is identical everywhere.

## 3. Rounding contract

Each path has a defined rounding behavior, **verified against source
2026-08-14** (rows state what the code does, per path, because two paths
over the same domain do not all share one rule — see the caveat below):

| Path | Multiply | Divide | Wide-tier downscale |
| ---- | -------- | ------ | ------------------- |
| Binary, imperative `FixedPoint` | **profile-dependent**: floor (realtime/compact), half-even (embedded), truncate toward zero (balanced/scientific) | truncate toward zero (all profiles) | round-to-nearest, ties toward +∞ |
| Binary, canonical/UGOD tier arithmetic | round-to-nearest, ties toward +∞ | round-half-away-from-zero | round-to-nearest, ties toward +∞ |
| Decimal, imperative `DecimalFixed<D>` | round-half-even (banker's) | round-half-even (banker's) | — |
| Decimal, canonical domain | exact (decimal places grow, no rounding) | truncate toward zero | (dp-cap rule pinned during unification) |
| Balanced ternary | truncate toward zero | truncate toward zero | — (transcendentals route via binary) |

The full per-site evidence is in
[docs/design/ROUNDING_CENSUS.md](docs/design/ROUNDING_CENSUS.md)
(2026-08-14): the imperative binary multiply was found to use *three*
different rules across profiles, and the imperative divide truncates on
every profile. This table states shipping behavior; the unification target
(one rule per domain on every path — binary nearest-ties-up, decimal
banker's, ternary nearest/tie-free) is ROADMAP 0.5.0 item 0c.

The rules differ by operation, by path, AND (imperatively) by profile —
historical artifacts, never retroactively unified. Consequence, measured
2026-08-14: **direct storage-tier multiply differs between the imperative
and canonical paths on ~half of all inexact products** (1 ulp, floor vs
nearest — 48.7% of 44,044 sampled pairs on compact), and divide diverges
analogously on every profile (truncate vs half-away). Path independence
currently holds for compound results but NOT for direct storage-tier
mul/div; unification (ROADMAP 0.5.0 item 0c — one rule per domain on
every path) repairs this. Each individual path remains fully
deterministic and bit-identical cross-platform. Every compound path —
every transcendental, dot product, decomposition, matrix chain, and fused
op — computes at tier N+1 (double the fractional bits) and rounds to
storage exactly once through the shared ties-up downscale; those results
are path-independent today and stay so. The reasoning is expanded in
[the precision guide](docs/README_PRECISION.md).

## 4. Cross-profile semantics

Different profiles have different bit-widths and therefore round to different
values; a Q32.32 result cannot hold Q64.64 digits. That divergence is inherent.

- **Guaranteed today**: determinism *within* a profile (§2).
- **Intended, not yet formally contracted**: the relationships *between* profiles
  — that widening a narrower value into a wider profile is exact (a pure bit
  shift), and that narrowing is a single correct rounding with no double-rounding
  artifact, and the serialization semantics for foreign-profile values. This is a
  tracked roadmap item (see [ROADMAP.md](ROADMAP.md), "cross-profile rounding
  contract"); do not rely on cross-profile bit relationships until it is stated
  and CI-gated.

Serialization (`imperative::serialization`) tags each value with its profile; a
reader encountering a foreign profile should treat the cross-profile rules above
as not-yet-guaranteed.

## 5. Precision & limits

Accuracy is defined by the test suite against mpmath references (50–250 digits,
embedded as exact strings), not by adjectives. The methodology and the honest
limits — input-representation error for values inexact in a pinned domain (the
canonical router auto-routes each literal to a domain where it is exact),
condition-number amplification for ill-conditioned systems, and the determinism of
whatever error remains — are documented in
[the precision guide](docs/README_PRECISION.md). Both user-reachable decimal paths
(imperative `DecimalFixed` and canonical `gmath()`) are graded in CI against
[mootable/decimal-scaled](https://github.com/mootable/decimal-scaled)'s independent
adversarial corpus.

## 6. Requirements & dependencies

- **Edition** 2021. No MSRV is currently pinned.
- **Default build has zero runtime dependencies.** Lookup tables are generated at
  build time by a pure-Rust `build.rs` (build-dependencies only) and are checked
  in, so the default build does not regenerate them.
- **Optional runtime dependencies**, each behind a feature: `infinite-precision`
  (`num-bigint`/`num-traits`/`num-integer`, for the BigInt symbolic tier), `serde`
  (`serde`), `inference` (`rayon`, for parallel TQ1.9 matvec).
- **Profile selection** is compile-time, via `GMATH_PROFILE` (or the matching
  Cargo feature). Switching profiles requires clearing `target/*/incremental/`.

## 7. What gMath consumes

Nothing at runtime by default — it is a pure computation library with no I/O, no
network, and no ambient state beyond a thread-local evaluator workspace. It
produces values; it does not call out.

## Disclaimer

This software is provided **"as is"**, without warranty of any kind, express or
implied. Use of this software is entirely at your own risk. In no event shall the
author or contributors be held liable for any damages arising from the use or
inability to use this software.

---

Built by **Niels Erik Toren** — [support & donations](README.md#author--support).
