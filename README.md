# g_math

Pure-Rust, zero-float, deterministic multi-domain fixed-point arithmetic.

Built by **Niels Erik Toren** · published as `g_math` on [crates.io](https://crates.io/crates/g_math)

---

> **0.7.0: the format is a type.** `Fixed<S, F>` (or `Q16_16` .. `Q256_256`)
> names the format of each value, every format is available in every build,
> and `GMATH_PROFILE` / `GMATH_FRAC_BITS` are gone. The transcendentals,
> canonical chains and closed-form geometry are correctly rounded over their
> full range. The ternary inference kernels (`tq19`) are the separate crate
> `g_math_inference`. Migration in about three edits per program:
> **[docs/UPGRADING_0.7.md](docs/UPGRADING_0.7.md)**; every change:
> [CHANGELOG.md](CHANGELOG.md).
>
> **Earlier**: 0.5.0 unified rounding (some final digits moved), 0.6.0 added
> certified arithmetic and exact predicates, 0.6.4 made operators and
> narrowings fail loudly and every multi-step computation round once, 0.6.5
> to 0.6.7 added the decimal tie rule, the wide-tier gates and caller
> buffers without changing in-range results, and 0.6.8 deprecated `tq19`.

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
rational), and 17 correctly rounded transcendental functions plus linear algebra,
geometry and certified arithmetic are built on top. Every format, from
`Fixed<i32, 2>` to Q256.256, is a type and any number of them coexist in one
program. It is one component of the Geodineum ecosystem and is consumed by its
geometric and inference projects.

## Public build surface

The build-with surface is five layers; everything beneath them is internal.

- **Formats** - `g_math::fixed_point::{Fixed, Q16_16, Q32_32, Q64_64,
  Q128_128, Q256_256}`: `Fixed<S, F>` over the storage words `i32` (any
  `2 <= F <= 30`), `i64`, `i128`, `I256`, `I512`; the traits `FixedNum`,
  `Storage`, `WideInt` and `Scalar` describe a format to generic code.
- **Canonical / router** - `g_math::canonical`: `gmath("…")` builds a lazy
  expression, `evaluate::<Q64_64>(…)` runs it in a format with cross-domain
  routing and one rounding for the whole tree.
- **Imperative** - `Fixed<S, F>`, `FixedVector<T>`, `FixedMatrix<T>` and
  `DecimalFixed<DECIMALS>`: direct `Copy` types, one call is one engine
  invocation.
- **Fused** - `g_math::fixed_point::imperative::fused`: whole accumulation patterns
  at the wide compute tier (norms, softmax and its mixes, `rms_norm`,
  `sigmoid_mul`, `entropy`, the correctly rounded quadratic form), several with
  forms that write into caller-provided slices.
- **Geometric** - `imperative::{decompose, derived, matrix_functions, manifold,
  lie_group, curvature, projective, fiber_bundle, ode, tensor, tensor_decompose,
  serialization}`.
- **Wide** - `g_math::wide`: Q64.64 `exp`/`ln`/`sin`/`cos`, `sigmoid`/`softplus`/
  `silu`, a correctly rounded square root, π constants, an exact
  decimal-literal parser, exact decimal ratios and a full-width integer
  `mul_div`, for quantities outside the storage tiers. Each function's
  measured error is in the module documentation.
- **Compute tier** - `g_math::compute_tier`: the tier-N+1 engines over a
  format's compute word (`Compute<T>`, `2F` fraction bits).
- **Ternary inference** - the TQ1.9 weight formats and kernels (`g_math::tq19`
  behind the `inference` feature until 0.6.7) are the separate crate
  `g_math_inference` from 0.7.0.
- **Internal** - the `universal`/`fasc` evaluator, the domain implementations
  beneath `DecimalFixed` and the ternary types, and the shadow/router internals are not part of the surface and may change between
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
| [Imperative](docs/README_IMPERATIVE.md) | `Fixed<S, F>`/`FixedVector`/`FixedMatrix` + `DecimalFixed`: direct, no-routing, `Copy` types for hot loops |
| [Transcendentals](docs/README_TRANSCENDENTALS.md) | The 17 functions: engines, correct rounding, fused `sincos`/`sinhcosh`, `try_*` variants |
| [Fused operations](docs/README_FUSED.md) | Compute-tier `sqrt_sum_sq`, `euclidean_distance`, `softmax`, `softmax_mix`, `rms_norm_factor`, `silu` |
| [Linear algebra](docs/README_LINALG.md) | LU/QR/Cholesky/SVD/eigen/Schur, solvers, matrix functions |
| [Geometry](docs/README_GEOMETRY.md) | Manifolds, Lie groups, curvature, projective geometry, fiber bundles, ODE, tensors |
| [Balanced ternary](docs/README_TERNARY.md) | Balanced-ternary arithmetic and trit packing |
| [Formats & precision](docs/README_PRECISION.md) | The formats, choosing one, rounding, validation, feature flags, build |
| [Upgrading to 0.7](docs/UPGRADING_0.7.md) | From profiles to format types, step by step |

## Capabilities

- **Four numeric domains** - each value is held in the representation that is exact
  for it (binary, decimal, balanced ternary, or symbolic rational), routed
  automatically.
- **17 transcendentals, correctly rounded** - computed one tier wider than
  storage, with a sure path at about 1000 bits when the result lies near a
  rounding boundary, so every result is the nearest value of the format;
  reachable lazily (`gmath`), directly (`Fixed`), or natively in decimal
  (`DecimalFixed`). Canonical chains and the closed-form geometry maps are
  rounded once from their exact value too.
- **Two computation paths** - a routed canonical layer for mixed or chained work
  and a direct imperative layer for known-domain hot loops. Results are
  path-independent: one rounding rule per domain on every path (unified
  0.5.0, permanently test-gated: see [CONTRACT.md](CONTRACT.md) §3).
- **Certified arithmetic and exact predicates** (0.6.0) - `Interval` and
  `DecimalInterval` return a bracket that provably contains the exact result
  for `+ - * /`, square root, dot products and quadratic forms (0.6.1: the
  quadratic form's bracket is at most 1 ulp wide, and `fused::quadratic_form`
  is its correctly rounded scalar); `pd_verdict`
  proves a matrix positive definite (or not, or says it cannot decide and hands
  back the straddling bracket); `orient2d`, `orient3d`, `incircle`, `insphere`
  return an exact three-way sign. Transcendentals are deliberately excluded
  from the certified set until their bounds are proven. Design and measured
  widths: `docs/design/CERTIFIED_INTERVALS.md`.
- **Numerics on top** - fused ML ops, dense linear algebra, differential geometry
  and Lie groups.
- **Every format in one build** - from 2 to 256 fraction bits, chosen per
  value by its type; results are bit-identical across platforms for each
  format.
- **Zero float internally, zero runtime dependencies by default.**

## Contract

The precise contract - the supported surface and its stability boundary, the
determinism guarantee, the per-domain rounding rules, cross-format semantics, and
what the crate requires - is in **[`CONTRACT.md`](CONTRACT.md)**. Agents should
prime from **[`CONTRACT.scn.md`](CONTRACT.scn.md)**.

## Quick start

```bash
cargo add g_math
```

```rust
use g_math::canonical::{gmath, evaluate};
use g_math::fixed_point::{Fixed, Q64_64};

// The canonical layer: "100" and "50" route to binary, "/ 3" keeps the
// value exact via symbolic routing, and the result is rounded once.
let value = evaluate::<Q64_64>(&((gmath("100") + gmath("50")) / gmath("3"))).unwrap();
println!("{value}");

// The imperative layer: one type per format, any number per program.
type Q = Fixed<i32, 10>; // Q22.10
let x = Q::from_str("2").sqrt();
println!("{x}");
```

From here, follow the [guides](#guides): the [routing guide](docs/README_ROUTING.md)
for mixed-domain and chained work, or the [imperative guide](docs/README_IMPERATIVE.md)
for direct hot-loop arithmetic.

## Limits worth knowing

- **The binary domain can't hold every literal exactly.** `0.3` and `1/3` repeat
  in binary, so the direct `Fixed` path approximates them. The canonical
  `gmath(...)` router avoids this for you: it classifies each literal and routes it
  to a domain where it is exact (`0.3`→decimal, `1/3`→symbolic), carrying the exact
  value in a `CompactShadow` - the limit only applies when you pin the binary
  domain.
- **Ill-conditioned systems amplify input error.** Solver error scales with the
  matrix condition number; this is a property of finite precision, not of the
  implementation.
- **Precision differs by format.** A Q16.16 result cannot hold Q64.64 digits;
  choose the format for your accuracy and range needs (several in one program
  is fine).
- **Certified means the arithmetic, not your model.** A certified bracket
  encloses the exact result of the computation on the values you passed in; it
  cannot see rounding that happened while those values were produced. At
  Q16.16 the brackets of a 50-dimensional interval Cholesky are a few percent
  of the pivot; from Q32.32 upward they are negligible.
- **Correct rounding costs time where it is hard.** A result near a rounding
  boundary is decided at about 1000 bits; canonical chains are 1.2 to 1.9x
  slower than in 0.6.7 (3.4x at Q22.10) and wide-format `exp`/`ln`/`sin` add 12
  to 33 ns. Arithmetic and the kernels run at 0.6.7's speed.

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

**"Certified" is a technical term, not a warranty.** Where this documentation
says *certified*, *proven* or *enclosure*, it means exactly what
[CONTRACT.md](CONTRACT.md) §3 and `docs/design/CERTIFIED_INTERVALS.md` define:
a statement about the exact integer computation performed on the values passed
in, established by construction and gated by tests. It is not a warranty of
fitness for any purpose, and the paragraph above applies to it in full.

## License

Licensed under either of

* [Apache License, Version 2.0](LICENSE-APACHE)
* [MIT License](LICENSE-MIT)

at your option.
