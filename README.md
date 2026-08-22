# g_math

Pure-Rust, zero-float, deterministic multi-domain fixed-point arithmetic.

Built by **Niels Erik Toren** · published as `g_math` on [crates.io](https://crates.io/crates/g_math)

---

## What it is

`g_math` computes with scaled integers only. No `f32`/`f64` appears anywhere in
the arithmetic, validation, or comparison paths (float conversions exist solely as
caller-convenience `from_f64`/`to_f64` wrappers). Because every operation is
integer arithmetic, results are bit-identical on every architecture - which makes
the crate suitable for blockchain consensus, financial auditing, and reproducible
scientific computation.

It is a library, not a service: pure computation, no I/O, no ambient state. Values
route across four numeric domains (binary, decimal, balanced ternary, symbolic
rational), and 18 transcendental functions plus linear algebra, geometry, and
ternary-quantized inference are built on top. It is one component of the Geodineum
ecosystem and is consumed by its geometric and inference projects.

## Public build surface

The build-with surface is five layers; everything beneath them is internal.

- **Canonical / router** - `g_math::canonical`: `gmath("…")` builds a lazy
  expression, `evaluate(…)` runs it with cross-domain routing and chain
  persistence.
- **Imperative** - `g_math::fixed_point::{FixedPoint, FixedVector, FixedMatrix}`
  and `DecimalFixed<DECIMALS>`: direct `Copy` types, one call is one engine
  invocation.
- **Fused** - `g_math::fixed_point::imperative::fused`: whole accumulation patterns
  at the wide compute tier.
- **Geometric** - `imperative::{decompose, derived, matrix_functions, manifold,
  lie_group, curvature, projective, fiber_bundle, ode, tensor, tensor_decompose,
  serialization}`.
- **TQ1.9** - `g_math::tq19` (feature `inference`): standalone ternary inference.
- **Internal** - the `universal`/`fasc` evaluator, the domain implementations
  beneath `DecimalFixed` and the ternary types, the wide-integer types, and the
  shadow/router internals are not part of the surface and may change between
  releases.

The complete per-symbol index - every public item and a one-line summary - is
**generated** from source into **[`PUBLIC_API.md`](PUBLIC_API.md)**
(`rustc -O scripts/gen-public-api.rs -o /tmp/gen-public-api && /tmp/gen-public-api`).
Live signatures are on [docs.rs](https://docs.rs/g_math). This README never
re-hosts either.

## Guides

Each layer and cross-cutting concept has a focused guide under `docs/`.

| Guide | What it covers |
|---|---|
| [Routing (FASC)](docs/README_ROUTING.md) | The `gmath()`/`evaluate()` layer: lazy trees, cross-domain routing, chain persistence (tier N+1), UGOD graceful overflow, the shadow system, mode routing |
| [Imperative](docs/README_IMPERATIVE.md) | `FixedPoint`/`FixedVector`/`FixedMatrix` + `DecimalFixed`: direct, no-routing, `Copy` types for hot loops |
| [Transcendentals](docs/README_TRANSCENDENTALS.md) | The 18 functions: dedicated engines, composed forms, fused `sincos`/`sinhcosh`, `try_*` variants |
| [Fused operations](docs/README_FUSED.md) | Compute-tier `sqrt_sum_sq`, `euclidean_distance`, `softmax`, `softmax_mix`, `rms_norm_factor`, `silu` |
| [Linear algebra](docs/README_LINALG.md) | LU/QR/Cholesky/SVD/eigen/Schur, solvers, matrix functions |
| [Geometry](docs/README_GEOMETRY.md) | Manifolds, Lie groups, curvature, projective geometry, fiber bundles, ODE, tensors |
| [Ternary & TQ1.9](docs/README_TERNARY.md) | Balanced-ternary arithmetic and the trit-plane inference weight formats |
| [Precision & profiles](docs/README_PRECISION.md) | The five profiles, `GMATH_FRAC_BITS`, rounding, validation, feature flags, build |

## Capabilities

- **Four numeric domains** - each value is held in the representation that is exact
  for it (binary, decimal, balanced ternary, or symbolic rational), routed
  automatically.
- **18 transcendentals** - computed one tier wider than storage and rounded once,
  reachable lazily (`gmath`), directly (`FixedPoint`), or natively in decimal
  (`DecimalFixed`).
- **Two computation paths** - a routed canonical layer for mixed or chained work
  and a direct imperative layer for known-domain hot loops. Results are
  path-independent, with one documented tie-rule exception on exact half-ULP
  storage-tier multiplies (see [CONTRACT.md](CONTRACT.md) §3).
- **Numerics on top** - fused ML ops, dense linear algebra, differential geometry
  and Lie groups, and standalone TQ1.9 ternary inference.
- **Five precision profiles** - Q16.16 through Q256.256, chosen at compile time;
  results are bit-identical across platforms within a profile.
- **Zero float internally, zero runtime dependencies by default.**

## Contract

The precise contract - the supported surface and its stability boundary, the
determinism guarantee, the per-domain rounding rules, cross-profile semantics, and
what the crate requires - is in **[`CONTRACT.md`](CONTRACT.md)**. Agents should
prime from **[`CONTRACT.scn.md`](CONTRACT.scn.md)**.

## Quick start

```bash
cargo add g_math
```

```rust
use g_math::canonical::{gmath, evaluate};

// "100" and "50" route to binary, "3" keeps the result exact via symbolic routing.
let value = evaluate(&((gmath("100") + gmath("50")) / gmath("3"))).unwrap();
println!("{value}");
```

From here, follow the [guides](#guides): the [routing guide](docs/README_ROUTING.md)
for mixed-domain and chained work, or the [imperative guide](docs/README_IMPERATIVE.md)
for direct hot-loop arithmetic.

## Limits worth knowing

- **The binary domain can't hold every literal exactly.** `0.3` and `1/3` repeat
  in binary, so the direct `FixedPoint` path approximates them. The canonical
  `gmath(...)` router avoids this for you: it classifies each literal and routes it
  to a domain where it is exact (`0.3`→decimal, `1/3`→symbolic), carrying the exact
  value in a `CompactShadow` - the limit only applies when you pin the binary
  domain.
- **Ill-conditioned systems amplify input error.** Solver error scales with the
  matrix condition number; this is a property of finite precision, not of the
  implementation.
- **Precision differs by profile.** A Q16.16 result cannot hold Q64.64 digits;
  choose the profile for your accuracy and range needs.
- **Switching profiles needs a clean incremental cache** (`rm -rf
  target/*/incremental/`), or stale `cfg` artifacts can crash the build.

## Collaborate

Contributions are welcome. Open issues and pick up work on the ecosystem board
at [geodineum.com](https://geodineum.com); issues tagged `good-first-issue` are
a good place to start.

- Fork, branch, and open a pull request against `main`.
- Any change to a wire contract must update **both** `CONTRACT.md` and
  `CONTRACT.scn.md` in the same commit.
- A change to a signed extension must be re-signed in the same commit.

## Author & support

Built by **Niels Erik Toren**.

If you want to support the work:

| Currency | Address |
|---|---|
| Bitcoin (BTC) | `bc1qwf78fjgapt2gcts4mwf3gnfkclvqgtlg4gpu4d` |
| Ethereum (ETH) | `0xf38b517Dd2005d93E0BDc1e9807665074c5eC731` / `nierto.eth` |
| Monero (XMR) | `8BPaSoq1pEJH4LgbGNQ92kFJA3oi2frE4igHvdP9Lz2giwhFo2VnNvGT8XABYasjtoVY2Qb3LVHv6CP3qwcJ8UnyRtjWRZ5` |

## Disclaimer

This software is provided **"as is"**, without warranty of any kind, express or
implied. Use of this software is entirely at your own risk. In no event shall the
author or contributors be held liable for any damages arising from the use or
inability to use this software.

## License

Licensed under either of

* [Apache License, Version 2.0](LICENSE-APACHE)
* [MIT License](LICENSE-MIT)

at your option.
